#!/usr/bin/env bash
set -euo pipefail
source /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/libero-plus/env.sh
exec /home/wangchaoyang/envs/LaWAM/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 runtime/train_ddp.py \
  --config configs/stage_I.json --output runs/stage_I_20260923 \
  --steps 16000 --batch-size 2 --accumulation 8 \
  --workers 2 --cpu-threads 4 \
  --save-every 1000 --eval-every 250 --eval-batches 64 --diagnostic-batches 8 \
  --keep-checkpoints 2 "$@"
