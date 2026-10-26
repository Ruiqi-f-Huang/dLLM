"""Save full training checkpoints at an explicit set of optimizer steps."""

from pathlib import Path

from lightning.pytorch.callbacks import Callback


class MilestoneCheckpoint(Callback):
    """Persist one checkpoint for each requested optimizer step.

    Step zero is deliberately omitted: fresh SFT uses the Base checkpoint as
    its step-zero model, so copying it would waste several gigabytes.
    """

    def __init__(self, milestones, dirpath, filename="step-{step:06d}.ckpt"):
        steps = tuple(sorted({int(step) for step in milestones}))
        if not steps or steps[0] <= 0:
            raise ValueError("milestones must contain positive optimizer steps")
        self.milestones = steps
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.saved = set()

    @property
    def state_key(self):
        return f"{type(self).__qualname__}:{','.join(map(str, self.milestones))}"

    def state_dict(self):
        return {"saved": sorted(self.saved)}

    def load_state_dict(self, state_dict):
        self.saved = {int(step) for step in state_dict.get("saved", [])}

    def on_train_start(self, trainer, pl_module):
        del pl_module
        if trainer.world_size != 1:
            raise ValueError("MilestoneCheckpoint currently requires one device.")
        if trainer.is_global_zero:
            self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del pl_module, outputs, batch, batch_idx
        step = int(trainer.global_step)
        if step not in self.milestones or step in self.saved:
            return
        path = self.dirpath / self.filename.format(step=step)
        trainer.save_checkpoint(path, weights_only=False)
        self.saved.add(step)

