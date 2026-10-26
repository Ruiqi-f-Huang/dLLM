"""Exercise persisted metrics with accumulation, validation and a partial window."""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from callbacks.sft_monitor import SFTMonitor
from trainer_base import TrainerBase
from metrics import Metrics


class RecordingCSVLogger(CSVLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logged_steps = []

    def log_metrics(self, metrics, step=None):
        self.logged_steps.append(step)
        super().log_metrics(metrics, step=step)


class Toy(L.LightningModule):
    on_train_epoch_start = TrainerBase.on_train_epoch_start
    on_train_epoch_end = TrainerBase.on_train_epoch_end
    on_validation_epoch_start = TrainerBase.on_validation_epoch_start

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.metrics = Metrics()
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.validation_preserved_train = False

    def _eval_mode(self):
        pass

    def training_step(self, batch, batch_idx):
        loss = self.weight.square()
        n = batch['attention_mask'].sum()
        self.metrics.update_train(loss.detach() * n, 0., n)
        self.log('trainer/loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        self.validation_preserved_train = self.metrics.train_nlls.nll.weight > 0
        self.log('val/nll', self.weight.square(), on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=.01)


class MonitorTests(unittest.TestCase):
    def test_nonfinite_gradient_is_reported_without_changing_amp_behavior(self):
        monitor = SFTMonitor()
        monitor.nonfinite_gradients = 0
        parameter = torch.nn.Parameter(torch.ones(1))
        parameter.grad = torch.tensor([float('inf')])
        optimizer = torch.optim.SGD([parameter], lr=.01)
        monitor.on_before_optimizer_step(None, None, optimizer)
        self.assertIsNone(monitor.grad_norm)
        self.assertEqual(monitor.nonfinite_gradients, 1)

    def test_accumulation_csv_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = RecordingCSVLogger(directory, name='metrics', flush_logs_every_n_steps=1)
            mirror = CSVLogger(directory, name='mirror', flush_logs_every_n_steps=1)
            wandb_logger = WandbLogger(project='s-flm-test', offline=True,
                                      save_dir=directory, log_model=False)
            self.addCleanup(wandb_logger.experiment.finish)
            rows = [{'input_ids': torch.tensor([1, 2, 0, 0]),
                     'attention_mask': torch.tensor([0, 1, 1, 1])} for _ in range(6)]
            loader = DataLoader(rows, batch_size=1)
            model = Toy()
            trainer = L.Trainer(accelerator='cpu', devices=1, max_epochs=1,
                                accumulate_grad_batches=2, logger=[logger, mirror, wandb_logger],
                                callbacks=[SFTMonitor(2)], log_every_n_steps=1,
                                enable_checkpointing=False, enable_progress_bar=False,
                                enable_model_summary=False, num_sanity_val_steps=0)
            trainer.fit(model, loader, loader)
            path = Path(logger.log_dir)
            summary = json.loads((path / 'summary.json').read_text())
            self.assertEqual(summary['final_step'], 3)
            self.assertEqual(summary['examples_seen_session'], 6)
            self.assertEqual(summary['supervised_tokens_seen_session'], 18)
            self.assertEqual(summary['code_eos_tokens_seen_session'], 6)
            records = [json.loads(line) for line in (path / 'monitor.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in records], [2, 3])
            self.assertTrue(all(math.isfinite(r['monitor/grad_norm_pre_clip']) for r in records))
            self.assertEqual(records[-1]['monitor/lr'], .01)
            self.assertTrue(model.validation_preserved_train)
            with (path / 'metrics.csv').open() as fp:
                csv_rows = list(csv.DictReader(fp))
            self.assertTrue(any(r.get('train/nll') for r in csv_rows))
            self.assertTrue(any(r.get('val/nll') for r in csv_rows))
            self.assertTrue(all(math.isfinite(float(r['train/nll'])) for r in csv_rows if r.get('train/nll')))
            with (Path(mirror.log_dir) / 'metrics.csv').open() as fp:
                mirrored = list(csv.DictReader(fp))
            self.assertTrue(any(r.get('monitor/lr') for r in mirrored))
            self.assertEqual(wandb_logger.experiment.summary['monitor/examples_seen_session'], 6)
            self.assertEqual(wandb_logger.experiment.summary['monitor/lr'], .01)
            self.assertEqual(logger.logged_steps, sorted(logger.logged_steps))
            self.assertEqual(records[-1]['monitor/optimizer_step'], 3)


if __name__ == '__main__':
    unittest.main()
