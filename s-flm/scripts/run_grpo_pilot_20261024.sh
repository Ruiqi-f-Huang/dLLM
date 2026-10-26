#!/usr/bin/env bash
set -euo pipefail
source /root/s-flm/activate.sh
RUN=/root/autodl-tmp/s-flm/outputs/grpo-pilot-20261024-01
trap 'echo "FAILED line $LINENO" > "$RUN/FAILED"' ERR
echo audit > "$RUN/STAGE"
while [[ ! -f "$RUN/audit/summary.json" ]]; do
  if ! kill -0 3244 2>/dev/null; then
    echo "Audit exited without summary" >&2
    exit 1
  fi
  sleep 10
done
python - <<'PY'
import json
from pathlib import Path
root=Path('/root/autodl-tmp/s-flm/outputs/grpo-pilot-20261024-01')
s=json.loads((root/'audit/summary.json').read_text())
assert s['groups']==256, s
if s['mixed'] < .1:
    raise RuntimeError('Less than 10% mixed-reward groups: inspect signal before training.')
train=[json.loads(x) for x in Path('data/gsm8k_pot_result/sft_train.jsonl').read_text().splitlines()]
valid=[json.loads(x) for x in Path('data/gsm8k_pot_result/sft_validation.jsonl').read_text().splitlines()]
assert not {r['sample_id'] for r in train}&{r['sample_id'] for r in valid}
assert not {r['question'].strip() for r in train}&{r['question'].strip() for r in valid}
print('Audit signal and train/validation split checks passed',s,flush=True)
PY
echo train > "$RUN/STAGE"
python -u scripts/train_mdlm_grpo.py \
  --checkpoint checkpoints/tinygsm/mdlm.ckpt --output "$RUN/train" \
  --limit 256 --updates 100 --iterations 2 --group-size 8 --microbatch 8 \
  --nfe 32 --temperature 0.3 --seed 42 --save-every 50 \
  > "$RUN/train.log" 2>&1
echo validation > "$RUN/STAGE"
python -u scripts/eval_grpo_pilot.py \
  --base checkpoints/tinygsm/mdlm.ckpt --rl "$RUN/train/step-000100.ckpt" \
  --output "$RUN/validation" --batch-size 32 > "$RUN/validation.log" 2>&1
echo complete > "$RUN/STAGE"
touch "$RUN/COMPLETE"

