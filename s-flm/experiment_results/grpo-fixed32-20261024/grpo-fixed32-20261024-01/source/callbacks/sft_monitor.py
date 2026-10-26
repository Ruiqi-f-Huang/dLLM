"""Single-device SFT diagnostics; independent of optional online services."""
import json
import math
import time
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback


class SFTMonitor(Callback):
    def __init__(self, every_n_steps=10):
        if every_n_steps < 1:
            raise ValueError('every_n_steps must be positive')
        self.every_n_steps = every_n_steps

    def on_train_start(self, trainer, pl_module):
        if trainer.world_size != 1:
            raise ValueError('SFTMonitor currently requires a single device.')
        self.started = time.perf_counter()
        self.initial_step = trainer.global_step
        self.last_step = trainer.global_step
        self.examples = self.tokens = self.target_tokens = self.content_tokens = 0
        self.train_seconds = 0.0
        self.window_seconds = 0.0
        self.window_examples = self.window_tokens = 0
        self.window_steps = 0
        self.grad_norm = None
        self.nonfinite_gradients = 0
        self.skipped_updates = 0
        self.scale_before_update = None
        self.lr = None
        self.path = Path(trainer.logger.log_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        if pl_module.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def _sync(self, module):
        if module.device.type == 'cuda':
            torch.cuda.synchronize(module.device)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._sync(pl_module)
        self.batch_started = time.perf_counter()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Lightning invokes this after AMP unscale and before gradient clipping.
        scaler = getattr(getattr(trainer, 'precision_plugin', None), 'scaler', None)
        self.scale_before_update = scaler.get_scale() if scaler is not None else None
        norms = [p.grad.detach().float().norm(2) for group in optimizer.param_groups
                 for p in group['params'] if p.grad is not None]
        self.grad_norm = float(torch.stack(norms).norm(2)) if norms else 0.0
        self.lr = float(optimizer.param_groups[0]['lr'])
        if not math.isfinite(self.grad_norm):
            self.nonfinite_gradients += 1
            self.grad_norm = None  # JSON null; preserve AMP's normal skip behavior.

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._sync(pl_module)
        elapsed = time.perf_counter() - self.batch_started
        ids, mask = batch['input_ids'], batch['attention_mask']
        examples, tokens = ids.shape[0], ids.numel()
        self.examples += examples
        self.tokens += tokens
        self.target_tokens += int(mask.sum())
        self.content_tokens += int((mask.bool() & (ids != pl_module.tokenizer.pad_token_id)).sum())
        self.train_seconds += elapsed
        self.window_seconds += elapsed
        self.window_examples += examples
        self.window_tokens += tokens
        if trainer.global_step > self.last_step:
            scaler = getattr(trainer.precision_plugin, 'scaler', None)
            if (scaler is not None and self.scale_before_update is not None
                    and scaler.get_scale() < self.scale_before_update):
                self.skipped_updates += 1
            self.window_steps += trainer.global_step - self.last_step
            self.last_step = trainer.global_step
            if trainer.global_step % self.every_n_steps == 0:
                self._emit(trainer, pl_module)

    def _emit(self, trainer, module):
        if not self.window_steps:
            return
        record = {
            'epoch': trainer.current_epoch,
            'monitor/optimizer_step': trainer.global_step,
            'monitor/lr': self.lr,
            'monitor/grad_norm_pre_clip': self.grad_norm,
            'monitor/nonfinite_gradient_updates_session': self.nonfinite_gradients,
            'monitor/amp_skipped_updates_session': self.skipped_updates,
            'monitor/successful_updates_session': trainer.global_step - self.initial_step - self.skipped_updates,
            'monitor/train_seconds_per_update': self.window_seconds / self.window_steps,
            'monitor/examples_per_second': self.window_examples / self.window_seconds,
            'monitor/padded_tokens_per_second': self.window_tokens / self.window_seconds,
            'monitor/examples_seen_session': self.examples,
            'monitor/supervised_tokens_seen_session': self.target_tokens,
            'monitor/code_eos_tokens_seen_session': self.content_tokens,
        }
        scaler = getattr(trainer.precision_plugin, 'scaler', None)
        if scaler is not None:
            record['monitor/amp_scale'] = scaler.get_scale()
        if module.device.type == 'cuda':
            record.update({
                'monitor/allocated_mib': torch.cuda.memory_allocated(module.device) / 2**20,
                'monitor/reserved_mib': torch.cuda.memory_reserved(module.device) / 2**20,
                'monitor/peak_allocated_mib': torch.cuda.max_memory_allocated(module.device) / 2**20,
            })
        # trainer.logger is only the first logger (CSV). Fan out diagnostics too.
        for logger in trainer.loggers:
            # Lightning logs an update at its zero-based index. Using the
            # completed-update count here would precede a lower native step.
            logger.log_metrics(record, step=max(0, trainer.global_step - 1))
            logger.save()
        with (self.path / 'monitor.jsonl').open('a') as fp:
            fp.write(json.dumps({'step': trainer.global_step, **record}, allow_nan=False) + '\n')
        self.window_steps = self.window_examples = self.window_tokens = 0
        self.window_seconds = 0.0

    def on_train_end(self, trainer, pl_module):
        self._emit(trainer, pl_module)
        summary = {
            'initial_step': self.initial_step, 'final_step': trainer.global_step,
            'examples_seen_session': self.examples,
            'padded_tokens_seen_session': self.tokens,
            'supervised_tokens_seen_session': self.target_tokens,
            'code_eos_tokens_seen_session': self.content_tokens,
            'nonfinite_gradient_updates_session': self.nonfinite_gradients,
            'amp_skipped_updates_session': self.skipped_updates,
            'successful_updates_session': trainer.global_step - self.initial_step - self.skipped_updates,
            'train_batch_seconds': self.train_seconds,
            'wall_seconds_including_validation': time.perf_counter() - self.started,
        }
        (self.path / 'summary.json').write_text(json.dumps(summary, indent=2))
