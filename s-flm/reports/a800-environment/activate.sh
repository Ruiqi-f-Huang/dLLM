#!/bin/bash
source /root/venvs/s-flm/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export HF_HOME=/root/autodl-tmp/s-flm/cache/huggingface
export HF_HUB_DISABLE_XET=1
export PIP_CACHE_DIR=/root/autodl-tmp/s-flm/cache/pip
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export SFT_CONFIG_NAME=sft-a800
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /root/s-flm
