#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1
mkdir -p robotwin/round4/runs
exec 9>robotwin/round4/runner.lock
flock -n 9 || exit 1
for variant in action_only future_aux; do
    export ROBOTWIN_ROUND4_VARIANT="$variant"
    run="robotwin/round4/runs/$variant"
    mkdir -p "$run"
    nice -n 10 ionice -c 3 /home/wangchaoyang/envs/LaWAM/bin/python -u -m robotwin.round4.train --variant "$variant" --output "$run" --steps 200 --accumulation 4 --validate-every 50 --validation-samples 16 >>"$run/train.log" 2>&1
    mkdir -p "robotwin/round4/assessment/$variant"
    nice -n 10 ionice -c 3 /home/wangchaoyang/envs/LaWAM/bin/python -u -m robotwin.round4.assess >>"robotwin/round4/assessment/$variant/run.log" 2>&1
done
