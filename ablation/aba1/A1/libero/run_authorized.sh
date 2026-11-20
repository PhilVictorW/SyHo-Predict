#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/ablation/aba1/A1/libero
export PYTHONDONTWRITEBYTECODE=1
/home/wangchaoyang/envs/LaWAM/bin/python -B launch.py --execute 2>&1 | tee -a launcher_gpu4x3_20261022.log
