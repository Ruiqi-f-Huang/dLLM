# MDLM diffu-GRPO pilot

Single CUDA device, native DiT LoRA, initialized from Base EMA. This is an
adaptation of d1's full-completion-mask probability proxy, not an exact
ancestral trajectory likelihood. Existing SFT training files are unchanged.

Run from the s-flm directory in its CUDA training environment:

```bash
# Check group diversity and mixed success/failure rewards before training.
python scripts/train_mdlm_grpo.py \
  --checkpoint checkpoints/tinygsm/mdlm.ckpt \
  --output outputs/grpo-audit-t03 --audit-only --limit 256 \
  --group-size 8 --microbatch 1 --temperature 0.3

# Short online pilot. Use a new output directory for every run.
python scripts/train_mdlm_grpo.py \
  --checkpoint checkpoints/tinygsm/mdlm.ckpt \
  --output outputs/grpo-pilot --updates 100 --iterations 2 \
  --group-size 8 --microbatch 1 --temperature 0.3
```

Default data is the existing `gsm8k_pot_result/sft_train.jsonl`. It must have
unique `sample_id`, `question`, and `answer` fields. Gold answers are reward-only;
reference code is never fed to the policy. Use `--data` for another training
partition; the script does not enforce split provenance, so do not pass validation
or test data. The selected shuffled training IDs are saved separately.

One rollout group contains one question and G candidates. `--updates` counts
optimizer steps, not generated groups. Each group is reused for `--iterations`
steps. All old/reference probabilities for those steps are cached before the
first update. Changing microbatch size changes the stochastic prompt-mask
layout when `--prompt-mask` is nonzero; keep it fixed within comparisons.

The current raw policy generates rollouts and receives updates. No EMA switching
occurs. Dropout is disabled for consistent probability ratios. Base EMA weights
remain frozen; temporarily zeroing LoRA B computes the reference. Advantages
subtract the group reward mean without dividing by standard deviation. Loss is
averaged over valid generated tokens across the entire group, including first
EOS and excluding PAD, MASK and tokens after EOS. No ELBO schedule weighting is
used. Prompt masking defaults to zero; `--prompt-mask 0.15` enables d1-style
prompt corruption. The default clipping epsilon is 0.2 and beta is 0.04; neither
is an established optimum for this model.

Rewards are binary execution correctness with the existing GSM8K evaluator's
code recovery and numeric comparison. Raw parsing is also logged. EOS-truncated
programs run in spawned processes with wall-clock/CPU/address-space/file-size
limits. This is not an OS security sandbox. A timeout or resource failure is a
zero reward. There is no format reward. Gold answers are not policy inputs.

Outputs:

- `run.json`, `config.yaml`, `selected_ids.json`: reproduction settings.
- `rollouts.jsonl`: question ID, candidate, generated token IDs/text, raw parse
  status and reward. Token IDs end before the first EOS.
- `metrics.jsonl`: per-group audit or per-update loss, proxy KL, clipping,
  pre-clipping gradient norm and rewards.
- `summary.json`: mean reward, pass-any, mixed-group and uniqueness fractions.
- `step-*.ckpt`: raw model, optimizer state, config and policy metadata.

Checkpoints have no EMA. Load them for evaluation with EMA disabled and the
saved LoRA configuration. Full training resume is not implemented: rerunning
from one of these checkpoints through `--checkpoint` is unsupported because
that argument expects a Base checkpoint with EMA. Checkpoint optimizer state
is retained for inspection/future resume support.

Evaluate at fixed 632-question validation, T=0.1, ancestral 32 NFE, at least two
seeds, matching hardware and question order. Compare raw RL parameters to the
Base EMA initialization and report wrong-to-right/right-to-wrong transitions.
Do not select hyperparameters on GSM8K Test or infer improvement from proxy loss
alone. All-identical-reward groups have zero policy advantage; KL can still
contribute after the policy diverges from the reference.

Local objective tests (no checkpoint or CUDA required):

```bash
python -m unittest discover -s tests -p test_diffu_grpo.py -v
```

CUDA integration, checkpoint loading, real sampling, and improvement in task
accuracy must still be validated in the training environment.
