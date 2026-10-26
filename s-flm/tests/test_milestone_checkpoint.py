"""Checkpoint milestone callback tests."""

import tempfile
import unittest
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from callbacks.milestone_checkpoint import MilestoneCheckpoint


class Toy(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        del batch, batch_idx
        return self.weight.square()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class MilestoneCheckpointTests(unittest.TestCase):
    def test_saves_only_requested_optimizer_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = MilestoneCheckpoint([1, 3], directory)
            trainer = L.Trainer(
                accelerator="cpu", devices=1, max_steps=4,
                logger=False, callbacks=[callback], enable_checkpointing=False,
                enable_progress_bar=False, enable_model_summary=False,
                num_sanity_val_steps=0)
            loader = DataLoader([torch.tensor(0.0)] * 4, batch_size=1)
            trainer.fit(Toy(), loader)
            files = sorted(path.name for path in Path(directory).glob("*.ckpt"))
            self.assertEqual(files, ["step-000001.ckpt", "step-000003.ckpt"])
            for filename, expected in zip(files, [1, 3]):
                checkpoint = torch.load(Path(directory) / filename,
                                        map_location="cpu", weights_only=False)
                self.assertEqual(checkpoint["global_step"], expected)

    def test_rejects_zero_milestone(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            MilestoneCheckpoint([0, 1], "checkpoints")


if __name__ == "__main__":
    unittest.main()
