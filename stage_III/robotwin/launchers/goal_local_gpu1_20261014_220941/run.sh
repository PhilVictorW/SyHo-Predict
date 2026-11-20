#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
export CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /home/wangchaoyang/envs/LaWAM/bin/python -u -m stage_III.robotwin.workflow --config /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/robotwin/launchers/goal_local_gpu1_20261014_220941/workflow.json --run-dir /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/robotwin/runs/goal_local_gpu1_20261014_220941 --gpu 1 --execute --final-evaluation > /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/robotwin/launchers/goal_local_gpu1_20261014_220941/console.log 2>&1
