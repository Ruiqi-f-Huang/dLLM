# MathInstruct -> GSM8K -> TinyGSM MDLM SFT feasibility audit

CPU-only data audit, no training or GPU model evaluation. All revisions and file hashes are in sources.json.

Results:
- MathInstruct total 262039; GSM8K PoT source data/PoT/gsm_gpt4.json: 14591 records.
- All 14591 match original GSM8K train under NFKC/whitespace normalization and recognized instruction suffixes, covering 6403 unique questions. No normalized test match. No full semantic dedup or pretraining-contamination audit claimed.
- Conservative AST conversion: 14328 records. All preserve numeric output; 7 disagree with gold under current scorer's strict tolerance.
- Verified and <=512 tokens including repeated-question docstring: 14309 records, 6316 questions.
- Quarantine 10 direct/aliased literal-return candidates, then AST dedup: 12785 variants; one per question: 6314 questions.
- Proposed validation: sample 200 original train IDs with random.Random(42) before filtering. 172 have program targets, all 200 remain generation validation questions. Remaining training pool: 6142 questions, or 12434 distinct program variants.
- One-program-per-question proposed training lengths min/median/max: 71/189/504 tokens.
- TinyGSM sample: first 1000 of first shard, not a random or complete audit. All have parsable simple_math_problem(), docstring and return, and fit 512 with this tokenizer.

Use the original question text and checkpoint separator. Separator is two literal characters, backslash+n, NOT a newline. Preserve answer-only mask with padding supervision. Do not truncate programs. Do not round or overwrite gold to rescue failures.

Candidate conversion uses AST: a single final print(expr) -> return expr, enclosed in simple_math_problem() with question docstring. No int casting. Imports, auxiliary functions/classes and complex structures excluded in initial conservative pool; not declared irrecoverable.

Execution audit uses four CPU processes, restricted builtins, 1GiB address-space limit and 1-second per-pair timeout; this is not a full OS sandbox.

Formal dataset freezing should add manual stratified code-quality review and optional stronger near-duplicate checks. The literal-return heuristic is not a proof of reasoning quality. Existing tiny_gsm loader hardcodes the original HF dataset; a separate local SFT data entry/config is needed.

These are audit candidates and proposed capacity/split measurements, not a completed SFT run or guaranteed performance improvement.
