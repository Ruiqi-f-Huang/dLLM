# Frozen final GSM8K Test comparison

Three predetermined models: Base EMA, semantic-weighted SFT711 EMA, positive32
raw step200. All1319 GSM8K Test questions, T=.1, ancestral32 NFE, FP32, seeds1/2,
batch8 over questions, matching per-batch random seed across models. Total7914
outputs, no pass@16. Preserve per-seed results, pair question bootstrap over the
mean of two seeds, raw/recovery execution, infrastructure retries and GPU cost.
A seed is not an additional independent test question. Test was previously used
for exploratory comparisons; disclose that history. No model reselection.

Script scripts/evaluate_final_gsm8k_test.py requires all three checkpoints before
creating an evaluation. --preflight-only checks required files and SFT metadata.
SFT expected SHA256:5b2f034c078a7557814cb9d243387ca525f9cb418568556740a569a23427029a,
global_step711, EMA updates711,96 tensors. Explicitly copy EMA before disabling
gradients; verify copied tensors, avoiding accidental raw SFT evaluation.

Server53432 currently has matching Base and positive32 checkpoints plus the full
Test dataset. Required SFT path is absent:
/root/autodl-tmp/s-flm/outputs/semantic-weighted-20261023-01/train/checkpoints/step-000711.ckpt.
Known source server54592 and server18707 refused SSH on this check. Uniform SFT200
on server53432 is not an authorized substitute. Do not retrain or replace SFT711
silently. Evaluation not started pending restoration of the specified artifact.

Example (after restoring SFT):
source /root/s-flm/activate.sh
python -u scripts/evaluate_final_gsm8k_test.py --base /root/s-flm/checkpoints/tinygsm/mdlm.ckpt --sft /root/autodl-tmp/s-flm/outputs/semantic-weighted-20261023-01/train/checkpoints/step-000711.ckpt --positive /root/autodl-tmp/s-flm/outputs/grpo-positive32-20261024-01/positive32-step200.ckpt --test /root/s-flm/data/gsm8k_test.json --output /root/autodl-tmp/s-flm/outputs/final-test-20261025-01

Update2026-10-26: source54592 became reachable. Required SFT711 was copied to
server20035 at the original path with matching archived SHA256. All three model
files and Test passed --preflight-only. CPU model load verified EMA updates711,
96 matching tensors, and exact EMA copy. Environment readiness is recorded in
reports/final_test_server_20035_environment_20261026.md. Earlier missing-file notes
above are historical. Evaluation has not been launched.

Launch update2026-10-26: user authorized execution on server20035. Started with
nohup, PID2567, output /root/autodl-tmp/s-flm/outputs/final-test-20261026-01,
log /root/autodl-tmp/final-test-20261026-01.log. Base seed1 completed its first
eight outputs. Runtime checks enforce identical tokenizer/prefixes and ancestral
sampling flags across models, full32 actual NFE, eps1e-5, unchanged prefixes,
and actual FP32 backbone output. Startup confirmation does not establish final
benchmark results. SSH disconnected after checking startup; no continued monitoring.
