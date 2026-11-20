#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export CUDA_VISIBLE_DEVICES="$(/home/wangchaoyang/envs/LaWAM/bin/python -c 'import json; c=json.load(open("config.json")); print(",".join(str(g) for g in c["physical_gpus"]))')"
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export TORCH_HOME="$SCRIPT_DIR/cache/torch" TRITON_CACHE_DIR="$SCRIPT_DIR/cache/triton"
export CUDA_CACHE_PATH="$SCRIPT_DIR/cache/cuda" TORCHINDUCTOR_CACHE_DIR="$SCRIPT_DIR/cache/inductor"
export TMPDIR="$SCRIPT_DIR/t"
mkdir -p "$TMPDIR"
exec /home/wangchaoyang/envs/LaWAM/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=3 train.py "$@"
