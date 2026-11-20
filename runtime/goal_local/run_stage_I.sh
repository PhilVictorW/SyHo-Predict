#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
GL_PYTHON=/home/wangchaoyang/envs/LaWAM/bin/python
GL_RUN=runs/goal_local/stage_I_20260921
GL_PREFLIGHT=runs/goal_local/preflight_20260921
mkdir -p "$GL_RUN" "$GL_PREFLIGHT"
trap 'code=$?; printf "Launcher exit code: %s\n" "$code" >> "$GL_RUN/launcher.log"; if (( code != 0 )); then printf "Training stopped; check train.log / preflight log.\n" >> "$GL_RUN/launcher.log"; fi' EXIT
# The complete optimizer preflight already passed. Do not repeat or mix its weights.
"$GL_PYTHON" -c 'import json;from pathlib import Path;s=json.loads(Path("runs/goal_local/preflight_20260921/status.json").read_text());assert s["phase"]=="completed", "Preflight has not passed"'
"$GL_PYTHON" -m goal_local.train --config configs/goal_local/stage_I.json --stage I \
  --output "$GL_RUN" --steps 10000 --batch-size 2 --accumulation 16 \
  --workers 4 --cpu-threads 4 --device cuda:0 --eval-batches 64 --diagnostic-batches 8 \
  --save-every 1000 --eval-every 250 --keep-checkpoints 2 --optimizer-device cpu \
  --save-initial > "$GL_RUN/train.log" 2>&1
