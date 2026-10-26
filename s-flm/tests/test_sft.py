"""Data boundary and fresh-finetune initialization checks (no downloads)."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

import dataloader
import main
from models.ema import ExponentialMovingAverage
from models.lora import apply_lora


class Tokenizer:
    name_or_path = 'byte-test'
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def __len__(self):
        return 259

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[b + 3 for b in text.encode()])


class ParameterModel(torch.nn.Module):
    def __init__(self, config, tokenizer=None):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        self._prepare_ema()

    def _get_parameters(self):
        return self.parameters()

    def _prepare_ema(self):
        self.ema = ExponentialMovingAverage(self.parameters(), self.config.training.ema)


class LoRAParameterModel(torch.nn.Module):
    def __init__(self, config, tokenizer=None):
        super().__init__()
        self.config = config
        self.backbone = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
        self.noise = torch.nn.Module()
        self.lora_modules = apply_lora(
            self.backbone, target_modules=['0'], rank=config.lora.rank,
            alpha=config.lora.alpha, dropout=config.lora.dropout)
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        self._prepare_ema()

    def _get_parameters(self):
        return self.parameters()

    def _prepare_ema(self):
        self.ema = ExponentialMovingAverage(self.parameters(), self.config.training.ema)


class SFTDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'train.jsonl'
        self.record = dict(sample_id='one', question='Question', code='return 3',
                           answer='GOLD_NOT_INPUT', reference_answer='#### 3')
        self.write()
        self.config = OmegaConf.create({
            'model': {'length': 64},
            'data': {'train_file': str(self.source), 'validation_file': str(self.source),
                     'cache_dir': str(self.root / 'cache'), 'wrap': False, 'streaming': False,
                     'separator': '\\n', 'train_on_prompt': False, 'train_on_pad': True},
        })
        self.tokenizer = Tokenizer()

    def write(self):
        self.source.write_text(json.dumps(self.record) + '\n')

    def test_prefix_eos_padding_and_no_gold_leak(self):
        row = dataloader.get_gsm8k_pot_dataset(self.config, self.tokenizer, 'train')[0]
        prefix = [1] + self.tokenizer('Question\\n').input_ids
        code = self.tokenizer('return 3').input_ids
        self.assertEqual(row['input_ids'][:len(prefix)].tolist(), prefix)
        self.assertEqual(row['input_ids'][len(prefix):len(prefix)+len(code)].tolist(), code)
        self.assertEqual(row['input_ids'][len(prefix)+len(code)].item(), 2)
        self.assertEqual(row['input_ids'][-1].item(), 0)
        self.assertEqual(row['attention_mask'][:len(prefix)].sum().item(), 0)
        self.assertTrue(row['attention_mask'][len(prefix):].all())
        self.assertTrue(torch.equal(row['attention_mask'], row['noise_mask']))
        self.assertEqual(len(row['input_ids']), 64)

    def test_cache_follows_contents_and_mask_settings(self):
        first = dataloader.get_gsm8k_pot_dataset(self.config, self.tokenizer, 'train')[0]
        self.record['code'] = 'return 4'
        self.write()
        second = dataloader.get_gsm8k_pot_dataset(self.config, self.tokenizer, 'train')[0]
        self.assertFalse(torch.equal(first['input_ids'], second['input_ids']))
        self.config.data.train_on_pad = False
        third = dataloader.get_gsm8k_pot_dataset(self.config, self.tokenizer, 'train')[0]
        self.assertEqual(third['attention_mask'][-1].item(), 0)
        self.assertEqual(third['noise_mask'][-1].item(), 1)

    def test_oversize_program_is_rejected(self):
        self.record['code'] = 'x' * 64
        self.write()
        with self.assertRaisesRegex(ValueError, 'refusing to truncate'):
            dataloader.get_gsm8k_pot_dataset(self.config, self.tokenizer, 'train')

    def test_same_question_cannot_cross_splits(self):
        with self.assertRaisesRegex(ValueError, 'overlapping question IDs'):
            dataloader.validate_gsm8k_pot_splits(self.config)

    def test_generation_validation_never_receives_reference_code(self):
        self.config.mode = 'gsm8k_eval'
        self.config.data.train = self.config.data.valid = 'gsm8k_pot'
        row = dataloader.get_dataset(self.config, self.tokenizer, 'valid')[0]
        self.assertEqual(row['input_ids'], [1] + self.tokenizer('Question\\n').input_ids)
        self.assertEqual(row['response_ground_truth'], '#### 3')
        self.assertNotIn('code', row)


class FinetuneTests(unittest.TestCase):
    def test_ema_or_raw_initialization_resets_history(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'base.ckpt'
            torch.save({'state_dict': {'weight': torch.ones(2, 2)},
                        'ema': {'shadow_params': [torch.full((2, 2), 2.)],
                                'num_updates': 250000, 'decay': .9999}}, path)
            for choice, expected in [('ema', 2.), ('raw', 1.)]:
                config = OmegaConf.create({'training': {'finetune_path': str(path),
                                                       'finetune_weights': choice, 'ema': .9}})
                model = main._initialize_finetune(ParameterModel, config, None)
                self.assertTrue(torch.equal(model.weight, torch.full((2, 2), expected)))
                self.assertTrue(torch.equal(model.ema.shadow_params[0], model.weight))
                self.assertEqual(model.ema.num_updates, 0)
                self.assertEqual(model.ema.decay, .9)
                self.assertIsNone(model.fast_forward_epochs)

    def test_incompatible_ema_fails(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'base.ckpt'
            torch.save({'state_dict': {'weight': torch.ones(2, 2)},
                        'ema': {'shadow_params': [torch.ones(3)]}}, path)
            config = OmegaConf.create({'training': {'finetune_path': str(path),
                                                   'finetune_weights': 'ema', 'ema': .9}})
            with self.assertRaisesRegex(ValueError, 'EMA parameters'):
                main._initialize_finetune(ParameterModel, config, None)

    def test_lora_initialization_imports_base_ema_and_resets_adapter_ema(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'base.ckpt'
            torch.save({
                'state_dict': {'backbone.0.weight': torch.ones(2, 2)},
                'ema': {'shadow_params': [torch.full((2, 2), 2.)],
                        'num_updates': 250000, 'decay': .9999}}, path)
            config = OmegaConf.create({
                'training': {'finetune_path': str(path),
                             'finetune_weights': 'ema', 'ema': .9999},
                'lora': {'enabled': True, 'rank': 1,
                         'alpha': 2, 'dropout': 0},
            })
            model = main._initialize_finetune(
                LoRAParameterModel, config, None)
            torch.testing.assert_close(
                model.backbone[0].weight, torch.full((2, 2), 2.))
            torch.testing.assert_close(
                model.backbone[0].lora_B, torch.zeros(2, 1))
            self.assertEqual(model.ema.num_updates, 0)
            # EMA follows only trainable parameters: LoRA A and B.
            self.assertEqual(len(model.ema.shadow_params), 2)


if __name__ == '__main__':
    unittest.main()
