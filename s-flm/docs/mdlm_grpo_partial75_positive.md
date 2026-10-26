# Partial75 positive-only controlled fit

Entry: scripts/diagnose_grpo_partial75_positive.py. Paired with
scripts/diagnose_grpo_partial75_overfit.py: same Base initialization, deterministic
8-question/16-program subset, fixed training masks and four held-out masks, FP32,
AdamW LR3e-5, 200 steps, norm cap .2, no KL/clipping. Only objective change:
success advantage remains +.5; failure advantage changes from -.5 to zero.
Keep the original 16-program denominator to preserve the success term's scaling.
Failures remain evaluation records and contribute no optimization gradient.

No new generation or generalization evaluation. Independent masks refer to the
same programs. Scores at 0/1/2/10/50/100/200 report train and held-out success,
failure, and gap separately. Reference geometry is diagnostic only. Frozen
parameter hashes and final adapter artifacts follow the signed-fit protocol.
