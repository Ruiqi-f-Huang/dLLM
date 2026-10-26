"""Mask invariants, source-to-token alignment, and MDLM objective integration."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from lift_masking import code_region, prepare_masks, select_mask, mdlm_lift_nll
import importlib.util


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': list(range(len(text))),
                'offset_mapping': [(i, i + 1) for i in range(len(text))]}


class LiftTests(unittest.TestCase):
    def test_unicode_docstrings_and_ordinary_strings(self):
        code = 'def solve():\n    """小明 buys apples."""\n    x = "keep"\n    return 2\n'
        _, region, ok = code_region(code, CharacterTokenizer())
        self.assertTrue(ok)
        self.assertFalse(any(region[code.index('"""'):code.rindex('"""') + 3]))
        self.assertTrue(all(region[code.index('"keep"'):code.index('"keep"') + 6]))
        self.assertTrue(region[code.index('2')])

    def test_invalid_python_falls_back(self):
        _, region, ok = code_region('def broken(', CharacterTokenizer())
        self.assertFalse(ok)
        self.assertFalse(any(region))

    def test_missing_offsets_fall_back(self):
        class SlowTokenizer:
            def __call__(self, text, **kwargs):
                if kwargs.get('return_offsets_mapping'):
                    raise NotImplementedError('Slow tokenizer')
                return {'input_ids': [1, 2]}
        ids, region, ok = code_region('x = 2', SlowTokenizer())
        self.assertEqual(ids, [1, 2])
        self.assertEqual(region, [False, False])
        self.assertFalse(ok)

    def test_boundary_crossing_bpe_excluded(self):
        class MergedTokenizer:
            def __call__(self, text, **kwargs):
                return {'input_ids': [1], 'offset_mapping': [(0, len(text))]}
        _, region, ok = code_region('"doc"\nx = 2', MergedTokenizer())
        self.assertTrue(ok)
        self.assertEqual(region, [False])

    def test_directions_and_per_example_policy(self):
        candidate = torch.ones(3, 4, dtype=torch.bool)
        scores = torch.tensor([[.1, .4, .2, .3]]).expand(3, -1)
        random = torch.tensor([[.2, .1, .4, .3]]).expand(3, -1)
        selected, modes = select_mask(candidate, torch.tensor([2, 2, 2]),
                                      scores, random,
                                      torch.tensor([[.1], [.5], [.9]]), 'lift3')
        self.assertEqual(modes.tolist(), [2, 0, 1])
        self.assertEqual(selected.tolist(), [[True, False, True, False],
                                             [False, False, True, True],
                                             [False, True, False, True]])

    def test_empty_zero_and_heterogeneous_k(self):
        candidate = torch.tensor([[0, 0, 0], [1, 0, 1], [1, 1, 1]], dtype=torch.bool)
        selected, _ = select_mask(candidate, torch.tensor([0, 1, 3]),
                                  torch.zeros(3, 3), torch.rand(3, 3),
                                  torch.tensor([[0.], [.5], [1.]]), 'lift3')
        self.assertEqual(selected.sum(-1).tolist(), [0, 1, 3])
        self.assertFalse((selected & ~candidate).any())

    def test_outside_regions_and_probability_endpoints(self):
        q = torch.tensor([[0.], [1.]])
        eligible = torch.tensor([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=torch.bool)
        noise = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=torch.bool)
        candidate, outside, k, _, rho = prepare_masks(
            q, eligible, noise, torch.Generator().manual_seed(1))
        self.assertEqual(k.tolist(), [0, 2])
        self.assertFalse(candidate[:, 0].any())
        self.assertFalse(outside[0].any())
        self.assertEqual(outside[1].tolist(), [False, False, False, True])
        self.assertTrue(((q + rho) <= 1).all())

    def test_two_forwards_restoration_rng_and_gradient(self):
        class Fake:
            config = SimpleNamespace(seed=1, lift=SimpleNamespace(policy='lift3'))
            global_step = 2
            global_rank = 0
            mask_index = 4
            def __init__(self):
                self.weight = torch.nn.Parameter(torch.tensor([0., 1., 2., 3., -100.]))
                self.calls = []
            def _sample_t(self, batch, step):
                return torch.full((batch,), .2)
            def noise(self, t):
                return -torch.ones_like(t), 1 - t
            def _sigma_from_alphat(self, alpha):
                return -alpha.log()
            def forward(self, x0, xt, sigma, context):
                self.calls.append((xt.clone(), torch.is_grad_enabled(), torch.rand(())))
                logits = self.weight.expand(*xt.shape, -1).clone()
                logits[..., self.mask_index] = -1e6
                logp = logits.log_softmax(-1)
                gold = torch.full_like(logp, -1e6).scatter(-1, xt[..., None], 0.)
                return torch.where((xt == self.mask_index)[..., None], logp, gold)
            def nll_per_token(self, logp, xt, x0, alpha, dalpha, **kwargs):
                return dalpha / (1 - alpha) * logp.gather(-1, x0[..., None]).squeeze(-1)
            log = Mock()
            neg_infinity = -1e6
        model = Fake()
        x = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 1]])
        valid = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1]])
        region = valid.bool().clone(); region[:, -1] = False
        torch.manual_seed(44)
        loss, _ = mdlm_lift_nll(model, x, None, 0, valid, region)
        proposal, scoring_grad, scoring_draw = model.calls[0]
        final, train_grad, train_draw = model.calls[1]
        self.assertFalse(scoring_grad); self.assertTrue(train_grad)
        self.assertEqual(scoring_draw, train_draw)
        self.assertEqual(final[0, 0], x[0, 0])
        self.assertTrue(((final == 4) <= (proposal == 4)).all())
        self.assertEqual(final[0, -1], proposal[0, -1])
        expected = -model.weight.log_softmax(-1)[x] * (final == 4) / .2
        torch.testing.assert_close(loss, expected)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(model.weight.grad).all())
        after_c = torch.get_rng_state()
        control = Fake()
        control.config = SimpleNamespace(seed=1, lift=SimpleNamespace(policy='random_restore'))
        torch.manual_seed(44)
        mdlm_lift_nll(control, x, None, 0, valid, region)
        torch.testing.assert_close(control.calls[0][0], proposal)
        torch.testing.assert_close(torch.get_rng_state(), after_c)

    @unittest.skipUnless(importlib.util.find_spec('lightning'), 'Full training environment required')
    def test_dataset_regions_and_cache_do_not_change_original_masks(self):
        import json
        import tempfile
        from pathlib import Path
        from omegaconf import OmegaConf
        from dataloader import get_gsm8k_pot_dataset

        class Encoded(dict):
            __getattr__ = dict.__getitem__
        class Tokenizer(CharacterTokenizer):
            name_or_path = 'character-test'
            bos_token_id, eos_token_id, pad_token_id = 1, 2, 0
            def __len__(self):
                return 512
            def __call__(self, text, **kwargs):
                return Encoded(super().__call__(text, **kwargs))
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'train.jsonl'
            code = 'def solve():\n    """question"""\n    return 2'
            source.write_text(json.dumps({'sample_id': 'one', 'question': 'Q', 'code': code}) + '\n')
            cfg = OmegaConf.create({'model': {'length': 128},
                'lift': {'enabled': True},
                'data': {'wrap': False, 'streaming': False,
                    'train_file': str(source), 'cache_dir': str(Path(root) / 'cache'),
                    'separator': '\n', 'train_on_prompt': False, 'train_on_pad': True}})
            row = get_gsm8k_pot_dataset(cfg, Tokenizer(), 'train')[0]
            prefix = 3  # BOS + Q + separator, each one character/token.
            self.assertTrue(row['lift_region_ok'])
            self.assertFalse(row['lift_region'][:prefix].any())
            self.assertTrue(row['lift_region'][prefix + len(code)])  # EOS
            self.assertFalse(row['lift_region'][prefix + len(code) + 1:].any())
            a = prefix + code.index('"""')
            b = prefix + code.rindex('"""') + 3
            self.assertFalse(row['lift_region'][a:b].any())
            cached = get_gsm8k_pot_dataset(cfg, Tokenizer(), 'train')[0]
            torch.testing.assert_close(row['lift_region'], cached['lift_region'])
            cfg.lift.enabled = False
            original = get_gsm8k_pot_dataset(cfg, Tokenizer(), 'train')[0]
            self.assertNotIn('lift_region', original)
            for key in ('input_ids', 'attention_mask', 'noise_mask'):
                torch.testing.assert_close(original[key], row[key])

    @unittest.skipUnless(importlib.util.find_spec('lightning'), 'Full training environment required')
    def test_validation_and_baseline_keep_original_path(self):
        from trainer_base import TrainerBase
        for train, policy in [(False, 'lift3'), (True, 'baseline')]:
            config = {'lift': {'enabled': True, 'policy': policy}}
            model = SimpleNamespace(config=config,
                _process_model_input=lambda x, valid: (x, None, valid),
                nll=Mock(return_value=(torch.ones(1, 3), None)),
                lift_nll=Mock(side_effect=AssertionError('must not run')))
            result = TrainerBase._loss(model, torch.ones(1, 3, dtype=torch.long),
                                       torch.tensor([[0, 1, 1]]), train_mode=train)
            self.assertEqual(result.loss.item(), 1.)
            model.nll.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec('lightning'), 'Full training environment required')
    def test_real_mdlm_lora_update_freezes_base(self):
        import os
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from algo import MDLM
        from models.lora import is_lora_parameter_name

        for name, fn in [('cwd', os.getcwd), ('device_count', lambda: 1),
                         ('eval', eval), ('div_up', lambda x, y: (x + y - 1) // y)]:
            if not OmegaConf.has_resolver(name):
                OmegaConf.register_new_resolver(name, fn)
        with initialize_config_dir(config_dir=os.path.abspath('configs'), version_base=None):
            cfg = compose(config_name='sft-lift-c')
        cfg.model.hidden_size = 32
        cfg.model.cond_dim = 16
        cfg.model.n_blocks = 1
        cfg.model.n_heads = 4
        cfg.model.length = 16
        cfg.model.dropout = 0.
        cfg.lora.rank = 2
        tokenizer = SimpleNamespace(vocab_size=32, mask_token=None, pad_token_id=0)
        for policy in ('random_restore', 'lift3'):
            with self.subTest(policy=policy):
                cfg.lift.policy = policy
                torch.manual_seed(13)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = MDLM(cfg, tokenizer).to(device)
                model.log = Mock()
                model._sample_t = lambda batch, step: torch.full((batch,), .2, device=device)
                # Synthetic pretrained-like weights avoid the deliberately
                # zero initialized DiT output/gates blocking adapter gradients.
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if not is_lora_parameter_name(name):
                            p.normal_(0, .1)
                frozen = {n: p.detach().clone() for n, p in model.named_parameters()
                          if not p.requires_grad}
                adapters = {n: p.detach().clone() for n, p in model.named_parameters()
                            if p.requires_grad}
                x = torch.randint(1, 32, (2, 16), device=device); x[:, 12:] = 0
                valid = torch.ones_like(x); valid[:, :4] = 0
                region = valid.bool(); region[:, 12:] = False
                # Match Lightning's outer mixed-precision context; no-grad
                # scoring must not poison its cached parameter casts.
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    result = model._loss(x, valid, noise_tokens=valid,
                                         train_mode=True, lift_region=region)
                self.assertTrue(torch.isfinite(result.loss))
                result.loss.backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in model.parameters() if p.requires_grad))
                optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
                optimizer.step()
                current = dict(model.named_parameters())
                for name, before in frozen.items():
                    torch.testing.assert_close(current[name], before, rtol=0, atol=0)
                self.assertTrue(any(not torch.equal(current[n], p) for n, p in adapters.items()))


if __name__ == '__main__':
    unittest.main()
