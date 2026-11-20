#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/wangchaoyang/envs/LaWAM/bin/python -u -m stage_III.libero_plus.full_train \
  --run /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/libero_plus/runs/grpo_full8380_20261013_010306 \
  --gpus 1,1,3,3,6,6,7,7 --resume 2>&1 | tee -a stage_III/libero_plus/full_train_8x2_retry_console.log
