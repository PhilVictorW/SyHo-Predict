#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
export CUDA_VISIBLE_DEVICES=5 CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/wangchaoyang/envs/LaWAM/bin/python -u -m robotwin.train \
  --config robotwin/configs/stage_I.json --stage I --output robotwin/runs/stage_I_20260924 \
  --steps 4000 --batch-size 1 --accumulation 32 --workers 2 --cpu-threads 4 \
  --save-every 500 --eval-every 250 --eval-batches 64 --diagnostic-batches 8 \
  --keep-checkpoints 8 --optimizer-device cpu --save-on-first-update
