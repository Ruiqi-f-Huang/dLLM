# Expanded positive32 coverage experiment

Single Base EMA arm: original32 fixed successful programs, FP32 LR3e-5, AdamW,
200 full-batch accumulated updates, microbatch2, cap .2, no KL/clipping. Same
positive-only program-mean objective as fixed8: success A=.5, failure A=0 with
original paired denominator. Failures are evaluated but have zero loss contribution.
Per-program exposure remains200; successful-program computation is four times
fixed8. Fixed75% train masks plus four independent fixed masks; scores at
0/1/2/10/50/100/200. Adapter artifacts at50/100/200; frozen parameter hash check.

Pipeline scripts/run_grpo_positive32.py freezes128 validation questions by hash
before training, deduplicates IDs and normalized text against the supplied full
training JSONL and fixed32. Validation remains within the existing local split;
prior experimental exposure and pretraining contamination are not ruled out.
Only step200 receives generation evaluation:32 training questions and128 validation,
16 draws/question at T=.3 (primary) and .1 (secondary), FP32/NFE32. Base validation
uses identical seeds; raw and recovery evaluator correctness are retained. Existing
Base and fixed8 results provide paired training-question comparisons (original8,
added24, all32). No validation-based checkpoint selection. COMPLETE/FAILED markers
and per-stage logs permit later inspection without supervision.

This is positive rollout distillation, not online GRPO. Coverage and total successful
program compute change together; no compute-matched second arm is run.
