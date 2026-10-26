# MDLM code-region LIFT pilot

This implements a **MDLM adaptation**, not a faithful full-region reproduction
of the LIFT repository. It does not change the inference sampler or claim
GSM8K improvements. The completed A800 pilot and its limits are recorded in
[the experiment report](../reports/gsm8k_lift_pilot_20261023.md).

## Method provenance

| Component | Source and behavior |
| --- | --- |
| Expanded candidate masks | LIFT `SFT/sft_trainer.py:get_rdro_rho`: sample extra corruption rho from `[0.1, 1-q]` if `1-q > 0.1`, otherwise `[0, 1-q]`. |
| Candidate size reduction | LIFT: retain `round(candidate_count * q/(q+rho))` masked positions. |
| Difficulty score | LIFT: current model's log probability of the gold token in the expanded-mask input, without gradients. |
| Time selection | LIFT-3 `time`: argmax over `[3q(1-q), q, 1-q]`, ordered random/easy/hard. This is a deterministic branch choice, not sampling a mixture. |
| Restore rejected positions | LIFT: restore gold tokens at rejected candidate positions, then make a second, trainable forward. No approximate variant. |
| Mask probability | MDLM adaptation: use `q=1-alpha(t)`, not raw time. Keep MDLM's existing time sampler. |
| Ranking scope | MDLM adaptation: non-docstring Python code plus EOS; docstrings and PAD retain ordinary random corruption and supervision. Prompt stays visible. |
| Batch handling | MDLM adaptation: choose a mode for each example using its own q, instead of LIFT's batch-mean time. |
| Zero retained positions | MDLM adaptation: permit k=0; do not force one target or renormalize by selected positions. |
| Loss | MDLM adaptation: retain `-alpha'(t)/(1-alpha(t))`, supervised-token denominator, and Base-aligned PAD treatment. No extra semantic weights. |
| Random streams | Experimental control: B/C have identical candidate RNG; both score under `fork_rng` so scoring dropout does not consume training RNG. Ordinary-mask draws match the global RNG consumption of A's `q_xt`. |
| Mixed precision | MDLM/Lightning engineering: disable AMP cast caching only during no-grad scoring, while retaining the surrounding dtype. This prevents detached cached casts from removing gradients in the trainable forward. |

Only continuous-time MDLM with `elbo`, `log_probs`, no time conditioning, a non-adaptive noise
schedule, the original time proposal, and conditional GSM8K PoT is supported
for this pilot. Importance sampling, time change of variables, high-noise
oversampling, and combined semantic weights are rejected.

Model-dependent mask selection changes the objective: training values logged
under existing NLL metric names are **not the original MDLM ELBO estimate**.
Validation always uses the original random-mask objective, including code/EOS
and PAD diagnostics. Evaluating validation does not invoke the scoring pass.

## Region alignment and fallback

`lift_masking.code_region` parses Python and finds actual module/function/class
docstrings through AST nodes. UTF-8 AST columns are converted to character
offsets before matching tokenizer offsets. A BPE overlapping a docstring is
excluded conservatively. Other string literals remain code.

`dataloader.py` constructs `[prompt=false] + code_region + [EOS=true] +
[PAD=false]`, without changing input IDs or the original loss/noise masks.
The cache fingerprint includes `REGION_VERSION` and whether region annotation
is enabled. Invalid Python or unsupported/unreliable offsets produce an
all-false region, including EOS: that example falls back to ordinary random
masking. Token ID disagreement is an error rather than silently changing the
target. Dataset rows expose `lift_region_ok` for preflight audits.

## Arms and execution

| Config | Arm |
| --- | --- |
| `sft-lift-a` | Original MDLM random-mask SFT; region labels do not affect training. |
| `sft-lift-b` | Expanded mask and random restoration; scores are computed but ignored. |
| `sft-lift-c` | Expanded mask and time-dependent easy/random/hard restoration. |

All inherit Base-aligned LoRA rank 8/alpha 16/dropout .05, LR `3e-6`, global
batch 8, BF16, original PoT data and Base EMA initialization. Workers are set
to zero. Each arm stops after 200 updates, saves checkpoints at 100/200, and
validates every 100 training batches. A single GPU and batch 8 mean these
intervals coincide. Overrides changing accumulation must account for this.

From the repository root in the prepared training environment:

```bash
python main.py --config-name sft-lift-a
python main.py --config-name sft-lift-b
python main.py --config-name sft-lift-c
# Or, run sequentially:
bash scripts/train/gsm8k/lift_compare.sh
```

Logs include rho, eligible/candidate/selected counts, target/actual code mask
rates, selected/candidate gold log probabilities, fallback fraction, and mode
fractions. Region-excluded positions are not ranked. No additional checkpoint
weights are introduced; the per-microbatch selector RNG is derived from the
training seed, global step, accumulation index, and rank. Exact end-to-end
resume still depends on the framework restoring its ordinary RNG and loader.

The A800 checkout requires synchronization of the newer local training code;
its old checkpoint config alone is insufficient. Use the composed configs,
including the pinned tokenizer revision. Do not launch against the old server
checkout without updating its dependencies/config/code consistently.

## Validation before real experiments

```bash
python -m unittest discover -s tests -p 'test_lift.py' -v
python -m unittest discover -s tests -p 'test_sft*.py' -v
python -m unittest discover -s tests -p 'test_lora.py' -v
```

Check region fallback counts on the real tokenizer/data, then run a short GPU
smoke test of B/C for finite loss/gradients, successful LoRA updates, unchanged
frozen weights, and batch-8 peak memory. CPU checks do not establish A800
memory usage or CUDA numerical behavior.

After training, compare Base and all arms on the fixed 632 validation
questions, EMA, ancestral, T=.1, 32 NFE, generation seeds 1/2. Use execution
accuracy and paired wrong-to-right/right-to-wrong counts as primary evidence.
Report raw parse/execution errors separately from evaluator-recovered results.
Do not select methods on Test or extend training solely because loss falls.

## Implementation verification (2026-10-23)

Local CPU verification used an isolated temporary environment (Python 3.14,
PyTorch 2.14.1, Lightning 2.5.1); this differs from the A800's Python 3.12 /
PyTorch 2.7 environment. Eleven LIFT tests cover regions/cache, policy and
count boundaries, restored inputs, RNG isolation, analytic loss/gradients,
unchanged baseline/validation dispatch, and real small MDLM+LoRA updates with
frozen-base equality in both B/C.

Together with the existing SFT/LoRA/semantic-weight tests, 33 tests passed;
one CUDA-only GradScaler test was skipped. The offline W&B monitor test
initially failed because sandbox rules prohibit local sockets; rerunning that
test outside the sandbox passed. A/B/C Hydra composition, Python compilation,
shell syntax, and diff whitespace checks also passed.

These were local implementation checks. The user subsequently authorized
upload and experiments; A800 execution and its AMP cache fix are recorded in
[the pilot report](../reports/gsm8k_lift_pilot_20261023.md). The real-model
regression and full-model smoke test now include Lightning's outer BF16 AMP
context, which the initial standalone smoke omitted.
