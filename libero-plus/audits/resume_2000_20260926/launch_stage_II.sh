#!/usr/bin/env bash
set -euo pipefail
source /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/libero-plus/env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/wangchaoyang/envs/LaWAM/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 runtime/train_stage_two.py \
  --teacher runs/stage_I_20260923/retained_checkpoints/step_005000 \
  --output runs/stage_II_20260925 --optimizer-dir /tmp/lawam_plus_stageII_optimizer_20260925 \
  --steps 10000 --batch-size 1 --accumulation 8 --save-every 500 --eval-every 500 --eval-batches 32 "$@"
