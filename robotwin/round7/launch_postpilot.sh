#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1
mkdir -p robotwin/round7/postpilot_eval
exec 9>robotwin/round7/postpilot_eval/runner.lock
flock -n 9 || exit 1
nice -n 10 ionice -c 2 -n 7 /home/wangchaoyang/envs/LaWAM/bin/python -u -m robotwin.round7.postpilot_eval >>robotwin/round7/postpilot_eval/eval.log 2>&1
