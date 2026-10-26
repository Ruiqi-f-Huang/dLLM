# Fixed Partial75 objective diagnostic

Run scripts/diagnose_grpo_partial75_objective.py with --base, --online-run,
--partial-run and a fresh --output. No optimizer steps or generation are run.

Compare Base EMA and partial75 step250 on the original 32 success/failure pairs,
with four fixed fresh 75% masks per program (256 inputs/model). Masks are sampled
on CPU and reused exactly across models. Prompt stays visible; post-EOS/PAD stays
masked. Score only masked valid completion positions, in true FP32.

Outputs: per-program scores, question-bootstrap success/failure and gap deltas,
fixed mask tensors/hash, adapter displacement and gradient vectors. Gradients
use signed token-mean log probability with A=reward-.5, normalized separately
within success/failure strata. Balanced gradient averages both strata equally.
This is the policy surrogate gradient at old=current, excluding clipping and KL;
it does not reproduce every online training group. Direction alignment against
the net adapter displacement is local first-order evidence, not causal evidence
about the entire optimizer trajectory. Four mask replicates are averaged before
question-level bootstrap. One seed and selected training questions limit claims.
