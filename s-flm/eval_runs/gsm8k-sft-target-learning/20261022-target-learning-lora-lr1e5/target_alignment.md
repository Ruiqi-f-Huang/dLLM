# GSM8K PoT target-structure alignment

Generated programs are compared with the reference code for each question. This measures code-form imitation, not mathematical equivalence: correct solutions may legitimately use different operators or intermediate variables. Generation is one sampled response per question (32 denoising steps, seed 1).

`strict generated parseable` means the full generated function parses as-is. `evaluator recovered parseable` mirrors the project's evaluator, which can cut trailing malformed output after an earlier valid prefix. `contract_pass` is measured on that evaluator-recovered code and means one top-level `simple_math_problem` function, an assignment to `result`, and exactly one `return result`. `operator_multiset_jaccard` compares arithmetic operator counts; 1 means the multisets match. Neither metric alone proves mathematical equivalence.

| Split | Model | N | Correct | Strict parseable | Evaluator recovered parseable | Contract pass | Mean op Jaccard | Mean target/generated assignments | Mean target/generated dependency depth |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train-subset | base_ema | 256 | 23 (8.98%) | 56.64% | 98.44% | 49.61% | 0.319 | 5.64 / 4.30 | 2.99 / 1.67 |
| train-subset | step100_ema | 256 | 21 (8.20%) | 55.86% | 98.44% | 48.05% | 0.326 | 5.64 / 4.28 | 2.99 / 1.70 |
| train-subset | step200_ema | 256 | 23 (8.98%) | 56.25% | 98.44% | 48.44% | 0.328 | 5.64 / 4.30 | 2.99 / 1.67 |
| train-subset | step711_ema | 256 | 19 (7.42%) | 53.12% | 98.83% | 42.97% | 0.319 | 5.64 / 4.12 | 2.99 / 1.57 |
| validation | base_ema | 632 | 54 (8.54%) | 56.80% | 97.15% | 47.78% | 0.326 | 5.49 / 4.29 | 2.99 / 1.78 |
| validation | step100_ema | 632 | 54 (8.54%) | 56.01% | 97.15% | 47.78% | 0.323 | 5.49 / 4.29 | 2.99 / 1.77 |
| validation | step200_ema | 632 | 55 (8.70%) | 56.96% | 97.15% | 48.89% | 0.335 | 5.49 / 4.32 | 2.99 / 1.80 |
| validation | step711_ema | 632 | 44 (6.96%) | 52.06% | 96.99% | 43.35% | 0.308 | 5.49 / 4.06 | 2.99 / 1.67 |

## Structural metrics split by answer correctness

| Split | Model | Answer group | N | Contract pass | Mean op Jaccard |
|---|---|---|---:|---:|---:|
| train-subset | base_ema | correct | 23 | 100.00% | 0.820 |
| train-subset | base_ema | wrong | 233 | 44.64% | 0.269 |
| train-subset | step100_ema | correct | 21 | 100.00% | 0.841 |
| train-subset | step100_ema | wrong | 235 | 43.40% | 0.280 |
| train-subset | step200_ema | correct | 23 | 100.00% | 0.855 |
| train-subset | step200_ema | wrong | 233 | 43.35% | 0.276 |
| train-subset | step711_ema | correct | 19 | 100.00% | 0.825 |
| train-subset | step711_ema | wrong | 237 | 38.40% | 0.278 |
| validation | base_ema | correct | 54 | 100.00% | 0.813 |
| validation | base_ema | wrong | 578 | 42.91% | 0.280 |
| validation | step100_ema | correct | 54 | 100.00% | 0.831 |
| validation | step100_ema | wrong | 578 | 42.91% | 0.275 |
| validation | step200_ema | correct | 55 | 100.00% | 0.857 |
| validation | step200_ema | wrong | 577 | 44.02% | 0.285 |
| validation | step711_ema | correct | 44 | 100.00% | 0.899 |
| validation | step711_ema | wrong | 588 | 39.12% | 0.264 |

On the validation split, every answer counted correct also passed this `result`/`return result` contract, but most contract-conforming programs were still numerically wrong. Correct answers have much higher operator-multiset overlap than wrong answers, as expected when the generated calculation plan matches the reference. At step 711 both the contract rate and overall operator overlap are slightly lower than Base on validation; the train subset also shows no structural or accuracy gain. This is preliminary evidence that 711-step SFT has not improved target imitation, while format conformity alone is not enough to fix arithmetic correctness.

Use the companion CSV to inspect individual failures. A low contract rate points to format imitation; a high contract rate with low accuracy and weak operator/dependency alignment points to calculation planning or operand/value errors; high structural similarity with wrong answers points further toward grounding/arithmetic errors. Inspect examples manually before assigning a cause.
