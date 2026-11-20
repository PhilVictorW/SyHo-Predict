#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1
RUN=robotwin/round3/runs/teacher_terminal_hold36_20260926
mkdir -p "$RUN"
exec 9>"$RUN/runner.lock"
flock -n 9 || exit 1
exec nice -n 10 ionice -c 3 /home/wangchaoyang/envs/LaWAM/bin/python -u -m robotwin.round3.train --output "$RUN" --steps 200 --accumulation 4 --validate-every 50 --validation-samples 16 >> "$RUN/train.log" 2>&1
