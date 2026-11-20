#!/usr/bin/env bash
export ABA_ROOT=/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2
cd "$ABA_ROOT"
export LIBERO_HOME="$ABA_ROOT/runtime/LIBERO"
export LIBERO_CONFIG_PATH="$ABA_ROOT/runtime/libero_config"
export LIBERO_PYTHON=/home/wangchaoyang/envs/LIBERO4LaWAM/bin/python
export STAR_VLA_PYTHON=/home/wangchaoyang/envs/LaWAM/bin/python
export PYTHONPATH="$ABA_ROOT:$LIBERO_HOME"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export DISABLE_LAWM=True
export GPU_ID=0 EVAL_GPU_ID=0 NUM_GPUS=8
export OUTPUT_ROOT="$ABA_ROOT/results/ablation"
export LIBERO_RUN_GROUP=inference_without_lawm_and_queries
export NUM_TRIALS_PER_TASK=50 NUM_WORKERS=1
export SAVE_VIDEOS=False SAVE_FIRST_SUCCESS_VIDEO=False SAVE_SIMILARITY_VIDEO=False
export LIBERO_RESUME_INCOMPLETE=True
export WORKER_RESULT_TIMEOUT_SEC=1200 SERVER_STARTUP_TIMEOUT_SEC=1200
export PORT_BASE=22940 PORT_LOCK_ROOT="$ABA_ROOT/runtime/port_locks"
export TMPDIR="$ABA_ROOT/runtime/tmp"
export XDG_CACHE_HOME="$ABA_ROOT/runtime/cache"
export MPLCONFIGDIR="$ABA_ROOT/runtime/cache/matplotlib"
export NUMBA_CACHE_DIR="$ABA_ROOT/runtime/cache/numba"
export TORCHINDUCTOR_CACHE_DIR="$ABA_ROOT/runtime/cache/torchinductor"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export CKPT_PATH="$ABA_ROOT/results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt"
