"""Analytic checks of conditional masking, MDLM loss and skipped AMP updates."""
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lightning as L
import torch

from algo import MDLM
from trainer_base import AbsorbingState, TrainerBase
from models.ema import ExponentialMovingAverage
from callbacks.sft_monitor import SFTMonitor
from lightning.pytorch.loggers import CSVLogger


class AlgorithmTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for GradScaler')
    def test_real_amp_overflow_skips_parameter_and_ema_update(self):
        class AMPModel(TrainerBase):
            def __init__(self):
                L.LightningModule.__init__(self)
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.ema = ExponentialMovingAverage([self.weight], .9)
                self.tokenizer = SimpleNamespace(pad_token_id=0)

            def _get_parameters(self):
                return [self.weight]

            def to(self, *args, **kwargs):
                return L.LightningModule.to(self, *args, **kwargs)

            def on_fit_start(self):
                pass

            def on_train_epoch_start(self):
                pass

            def on_train_epoch_end(self):
                pass

            def on_train_start(self):
                self.ema.move_shadow_params_to_device(self.device)

            def training_step(self, batch, batch_idx):
                return self.weight.square().sum()

            def on_after_backward(self):
                if self.global_step == 0:
                    self.weight.grad.fill_(float('inf'))

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=.01)

        with tempfile.TemporaryDirectory() as directory:
            model = AMPModel()
            logger = CSVLogger(directory, name='amp-test')
            trainer = L.Trainer(accelerator='gpu', devices=1, precision='16-mixed',
                                max_steps=2, logger=logger, callbacks=[SFTMonitor(1)],
                                enable_checkpointing=False,
                                enable_model_summary=False, enable_progress_bar=False)
            rows = [{'input_ids': torch.ones(2, dtype=torch.long),
                     'attention_mask': torch.ones(2, dtype=torch.long)} for _ in range(2)]
            trainer.fit(model, torch.utils.data.DataLoader(rows, batch_size=1))
            self.assertEqual(trainer.global_step, 2)  # Lightning counts attempts.
            self.assertEqual(model.ema.num_updates, 1)
            torch.testing.assert_close(model.weight.detach().cpu(), torch.tensor([.98]))
            summary = json.loads((Path(logger.log_dir) / 'summary.json').read_text())
            self.assertEqual(summary['amp_skipped_updates_session'], 1)
            self.assertEqual(summary['successful_updates_session'], 1)

    def test_prompt_stays_clean_and_targets_are_noised(self):
        model = SimpleNamespace(mask_index=4)
        x = torch.tensor([[0, 1, 2, 3]])
        valid = torch.tensor([[0, 0, 1, 1]])
        noisy = AbsorbingState.q_xt(model, x, torch.zeros(1, 1), False, valid)
        self.assertEqual(noisy.tolist(), [[0, 1, 4, 4]])
        clean = AbsorbingState.q_xt(model, x, torch.ones(1, 1), False, valid)
        self.assertTrue(torch.equal(clean, x))

    def test_mdlm_weight_and_gradient_match_analytic_answer(self):
        model = SimpleNamespace(mask_index=4, neg_infinity=-1000000.)
        logits = torch.zeros(1, 4, 5, requires_grad=True)
        xt = torch.tensor([[0, 1, 4, 4]])
        x0 = torch.tensor([[0, 1, 2, 3]])
        logp = MDLM._process_model_output_log_probs(model, logits + 0., xt, None)
        loss = MDLM.nll_per_token(model, logp, xt, x0,
                                 torch.tensor([[.5]]), torch.tensor([[-1.]]))
        expected = 2 * torch.log(torch.tensor(4.))
        torch.testing.assert_close(loss[0, :2], torch.zeros(2))
        torch.testing.assert_close(loss[0, 2:], expected.expand(2))
        loss.mean().backward()
        torch.testing.assert_close(logits.grad[0, :2], torch.zeros(2, 5))
        self.assertGreater(logits.grad[0, 2:].abs().sum().item(), 0)
        self.assertEqual(logits.grad[..., 4].abs().sum().item(), 0)

    def test_loss_normalizes_by_supervised_tokens_including_padding(self):
        model = SimpleNamespace(
            _process_model_input=lambda x, mask: (x, None, mask),
            nll=lambda *a, **kw: (torch.tensor([[99., 2., 4., 6.]]), None))
        loss = TrainerBase._loss(model, torch.zeros(1, 4, dtype=torch.long),
                                 torch.tensor([[0, 1, 1, 1]]))
        self.assertEqual(loss.num_tokens.item(), 3)
        self.assertEqual(loss.nlls.item(), 12)
        self.assertEqual(loss.loss.item(), 4)

    def test_code_and_padding_diagnostics_are_token_weighted(self):
        logged = {}
        model = SimpleNamespace(
            config={'sft_monitoring': True}, tokenizer=SimpleNamespace(pad_token_id=0),
            log=lambda key, value, **kwargs: logged.update({key: (float(value), kwargs['batch_size'])}),
            _process_model_input=lambda x, mask: (x, None, mask),
            nll=lambda *a, **kw: (torch.tensor([[99., 2., 4., 6.]]), None))
        TrainerBase._loss(model, torch.tensor([[1, 2, 3, 0]]), torch.tensor([[0, 1, 1, 1]]))
        self.assertEqual(logged['val/code_eos_nll'], (3., 2))
        self.assertEqual(logged['val/pad_nll'], (6., 1))

    def test_ema_does_not_advance_when_scaler_skips_update(self):
        class Model(TrainerBase):
            def __init__(self, scaler):
                L.LightningModule.__init__(self)
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.ema = ExponentialMovingAverage([self.weight], .9)
                self._trainer = SimpleNamespace(precision_plugin=SimpleNamespace(scaler=scaler))

            def _get_parameters(self):
                return [self.weight]

        for before, after, expected in [(16., 8., 0), (16., 16., 1), (16., 32., 1)]:
            scaler = SimpleNamespace(get_scale=lambda: before)
            model = Model(scaler)
            def step(*args, **kwargs):
                scaler.get_scale = lambda: after
            with patch.object(L.LightningModule, 'optimizer_step', step):
                model.optimizer_step()
            self.assertEqual(model.ema.num_updates, expected)


if __name__ == '__main__':
    unittest.main()
