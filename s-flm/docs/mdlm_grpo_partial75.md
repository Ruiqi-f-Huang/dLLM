# 75% Partial-MASK GRPO

Single new arm: Base EMA initialization, original fixed 32 questions, FP32,
LR 3e-5, 250 optimizer updates, group 8, two PPO iterations, rollout T=.3/NFE32,
beta .04 and clip .2. Existing full-MASK FP32 run supplies the control.

`--completion-mask .75` selects round(.75*N) valid completion tokens per
candidate (minimum one). Only selected tokens enter policy and KL losses.
Unselected valid tokens remain visible; prompt stays visible and post-EOS/PAD
tail stays masked. Same mask is used for old/current/reference and both PPO
iterations; each new rollout group draws a new seed. The full-group microbatch
is required to make mask identity independent of call boundaries. This is a
teacher-forced proxy, not exact trajectory RL. Existing model is time-unconditioned;
noise input retains the original proxy convention.

Monitoring logs losses, ratios, KL, gradient geometry and parameter updates,
mask seed and selected token counts. Fixed-pair gradient diagnostics are disabled
for this arm; full-MASK fixed-pair scores remain available in post-training evaluation.

The pipeline automatically evaluates 16 draws/question at T=.3 and .1, compares
against Base and full-MASK250 using paired question bootstrap, then reuses the
previous 64 Base trajectories at steps 0/8/16/24/31. `rl250` in the reused trajectory
scorer denotes this candidate; run.json records the actual checkpoint path.
Only step250 receives full evaluation; step100 is retained as a checkpoint.
One training seed and selected training questions do not establish generalization.

Entry point: scripts/run_grpo_partial75.py --base CHECKPOINT --online-run ONLINE
--trajectory-run TRAJECTORY --output NEW_DIRECTORY.
