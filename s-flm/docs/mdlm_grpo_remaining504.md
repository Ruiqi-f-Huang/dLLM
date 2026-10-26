# Remaining504 validation comparison on A800

User explicitly skips Base evaluation. Compare existing positive8 and positive32
step200 models only, on all504 validation questions remaining after excluding
the previously used128 by ID and normalized text. Verify training ID/text
disjointness before generation. Freeze questions.jsonl and selection.json.
No assertion of no previous project or pretraining exposure.

Entry scripts/evaluate_grpo_remaining_validation.py: FP32, NFE32, T=.3 primary,
T=.1 secondary,16 draws/question, batch8, shared seed protocol20261025. No model
training, Base generation, or checkpoint selection. Total32256 new samples.
Retain raw/recovery execution details, infrastructure failures and paired
question bootstrap of positive32 minus positive8. This comparison addresses
relative performance; it cannot establish gains over Base on this question set.
Coverage and training compute differ between the two models.

Server port53432; experiment /root/autodl-tmp/s-flm/outputs/grpo-remaining504-20261025-01.
Evaluation COMPLETE/FAILED/STAGE are within its evaluation subdirectory.
