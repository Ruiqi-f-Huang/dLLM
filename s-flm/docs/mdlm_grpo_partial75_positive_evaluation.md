# Positive-only partial75 model evaluation

scripts/evaluate_grpo_partial75_positive.py restores the step200 adapter onto the
original Base EMA and verifies a standard checkpoint reload preserves every
adapter tensor. It runs no optimizer steps. Checkpoint conversion is local to
the server evaluation directory.

Generation: original fixed32, 16 draws/question at T=.3/.1, FP32, existing evaluator
and matching evaluation seeds. Reuse original Base results. Report all32, trained8,
and other24 separately with paired question bootstrap. Other24 are unused by this
fixed8 fit but selected by earlier diagnostics, not an independent validation set.

Trajectory: reuse original fixed64 Base trajectories at steps 0/8/16/24/31;
score both Base and candidate at T=1/.3, report subgroup comparisons too.
Scorer label rl250 refers to positive step200 here; exact checkpoint is recorded.
No claim of exact final-program likelihood or RL-policy visit distribution.
The pipeline writes COMPLETE or FAILED and retains source runner/configuration.
