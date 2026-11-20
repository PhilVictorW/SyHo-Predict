#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/../../../..:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_CACHE_PATH="$SCRIPT_DIR/cache/cuda" TORCHINDUCTOR_CACHE_DIR="$SCRIPT_DIR/cache/inductor"
export TORCH_HOME="$SCRIPT_DIR/cache/torch" TMPDIR="$SCRIPT_DIR/t"
mkdir -p "$TMPDIR"
exec /home/wangchaoyang/envs/LaWAM/bin/python -m a0_single_latent.parallel_evaluation "$@"
