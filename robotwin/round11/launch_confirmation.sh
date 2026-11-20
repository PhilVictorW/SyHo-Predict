#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
export CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1
mkdir -p robotwin/round11/confirmation_500
exec 9>robotwin/round11/confirmation_500/runner.lock
flock -n 9 || exit 1
exec nice -n 10 ionice -c 2 -n 7 /home/wangchaoyang/envs/LaWAM/bin/python -u -m robotwin.round11.confirm >>robotwin/round11/confirmation_500/eval.log 2>&1
