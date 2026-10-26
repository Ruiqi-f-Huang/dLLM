# Fixed validation coverage comparison and seed replication

scripts/compare_grpo_positive_coverage.py performs no optimization. Frozen128
validation bytes must match the previous positive32 validation manifest. At each
T=.3/.1, generate16 times/question, batch8, NFE32, FP32, existing raw/recovery
execution evaluator.

Original seed20261024: reuse Base and positive32 results, fill only positive8.
Fresh seed20261025: evaluate Base, positive8 and positive32 with identical seed
protocol. Save raw outputs, seeds, infrastructure failures, summaries, and paired
question-bootstrap differences for Base->8, Base->32, 8->32 on original, fresh,
and combined batches. Combine counts within question (32 draws/question), then
bootstrap questions. No checkpoint selection or training occurs.

Single checkpoint/fit and the same validation questions limit inference; a fresh
sampling batch is not a fresh held-out question set. Coverage and training compute
are confounded in the8-to32 comparison. COMPLETE/FAILED and STAGE support later
inspection without supervision.
