#!/usr/bin/env bash
# All generated files stay under A2, including short-path IPC files.
A2_WORK="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$A2_WORK"
mkdir -p "$A2_WORK/.t" "$A2_WORK/runtime_cache" "$A2_WORK/logs"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$A2_WORK:$A2_WORK/../../../..:${PYTHONPATH:-}"
export TMPDIR=/proc/self/cwd/.t
export XDG_CACHE_HOME="$A2_WORK/runtime_cache"
export CUDA_CACHE_PATH="$A2_WORK/runtime_cache/cuda"
export TRITON_CACHE_DIR="$A2_WORK/runtime_cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$A2_WORK/runtime_cache/inductor"
export MPLCONFIGDIR="$A2_WORK/runtime_cache/matplotlib"
export NUMBA_CACHE_DIR="$A2_WORK/runtime_cache/numba"
export HF_HOME="$A2_WORK/runtime_cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
