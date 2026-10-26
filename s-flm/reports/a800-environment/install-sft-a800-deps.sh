#!/bin/bash
set -euo pipefail
export PIP_INDEX_URL=https://pypi.mirrors.ustc.edu.cn/simple
export PIP_CACHE_DIR=/root/autodl-tmp/s-flm/cache/pip
export CUDA_HOME=/usr/local/cuda
export PATH=/root/venvs/s-flm/bin:/usr/local/cuda/bin:$PATH
python -c "import torch,numpy; assert torch.__version__.startswith('2.7.0'); assert numpy.__version__=='1.26.4'"
python -m pip install -r /root/setup-logs/requirements-sft-a800.txt -c /root/setup-logs/sft-a800-constraints.txt
python -m pip check
