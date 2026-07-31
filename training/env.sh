#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT=/root/autodl-tmp/text2sql_qwen3
export PATH=/root/miniconda3/bin:$PATH
export PIP_CACHE_DIR="$PROJECT_ROOT/cache/pip"
export HF_HOME="$PROJECT_ROOT/cache/hf"
export MODELSCOPE_CACHE="$PROJECT_ROOT/cache/modelscope"
export XDG_CACHE_HOME="$PROJECT_ROOT/cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
