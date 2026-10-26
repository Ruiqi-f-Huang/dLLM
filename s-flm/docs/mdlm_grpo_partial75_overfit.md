# Fixed-program partial75 overfit diagnostic

Entry: scripts/diagnose_grpo_partial75_overfit.py. Single Base EMA arm, FP32,
LR3e-5, 200 updates, original deterministic controlled subset of 8 questions,
16 success/failure programs. One fixed exact 75% training mask/program and four
independent held-out masks/program. Same programs in both sets: tests mask-state
transfer, not question/program generalization. Prompt stays visible; invalid tail
stays MASK. Only masked completion positions are scored.

Train the program-mean signed log-probability objective with A=reward-.5; no KL
or clipping. AdamW weight decay0, gradient norm cap .2. Not an online GRPO efficacy
test. The signed negative-example loss is unbounded, so gap improvement alone
must be interpreted together with success and failure scores separately.

Scores at 0/1/2/10/50/100/200 include per-program train-mask scores and four
held-out-mask score sets. Gradient geometry compares program, token and reference
objectives; reference is measured but not used for optimization. Frozen parameters
are hashed before/after. Final adapter weights and optimizer are retained in
final_adapters.pt (a diagnostic artifact, not a standard full Lightning checkpoint).
Raw FP32 output is checked during scoring. COMPLETE/FAILED mark termination.
