#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
GL_PYTHON=/home/wangchaoyang/envs/LaWAM/bin/python
GL_RUN=runs/goal_local/stage_II_teacher2000_20260921
trap 'code=$?; printf "Launcher exit code: %s\n" "$code" >> "$GL_RUN/launcher.log"' EXIT
exec "$GL_PYTHON" -m goal_local.train --config configs/goal_local/stage_II.json --stage II \
  --stage-one runs/goal_local/teachers/stage_I_step_002000 \
  --output "$GL_RUN" --steps 10000 --batch-size 2 --accumulation 16 \
  --workers 4 --cpu-threads 4 --device cuda:0 --eval-batches 64 --diagnostic-batches 0 \
  --save-every 1000 --eval-every 250 --keep-checkpoints 2 --optimizer-device cpu \
  --save-initial > "$GL_RUN/train.log" 2>&1
