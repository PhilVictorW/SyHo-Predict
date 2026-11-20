#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[Stage III][第二轮] 第一轮最终 GateActor + Adam 继续训练；8380组 × 4条轨迹；GPU0/1/2/7各两路。"
/home/wangchaoyang/envs/LaWAM/bin/python -u -m stage_III.libero_plus.full_train --run /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/libero_plus/runs/grpo_round2_full8380_20261018_211335 --gpus 0,0,1,1,2,2,7,7 --ports 29900,29901,29902,29903,29904,29905,29906,29907 --groups-per-update 32 --initial-actor /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/libero_plus/launchers/round2_full8380_20261018_211335/actor_round1_update0262.pt 2>&1 | tee -a /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/libero_plus/launchers/round2_full8380_20261018_211335/console.log
