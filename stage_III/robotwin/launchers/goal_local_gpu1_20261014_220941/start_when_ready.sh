#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
exec /home/wangchaoyang/envs/LaWAM/bin/python -u /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/robotwin/launchers/goal_local_gpu1_20261014_220941/resume_ready.py "$@"
