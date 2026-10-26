#!/usr/bin/env bash
# Run sequentially on one GPU. Optional Hydra overrides are shared by all arms.
set -euo pipefail
cd "$(dirname "$0")/../../.."
for arm in a b c; do
  python main.py --config-name "sft-lift-${arm}" "$@"
done
