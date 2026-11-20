#!/usr/bin/env bash
set -uo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
export PYTHONUNBUFFERED=1
export LIBERO_HOME=/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2/runtime/LIBERO
export LIBERO_CONFIG_PATH=/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2/runtime/libero_config
export PYTHONPATH=/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change:/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2/runtime/LIBERO
echo '[Stage III][第二轮] 从已归档的第一轮最终 Actor 和 Adam 状态继续训练。'
/home/wangchaoyang/envs/LaWAM/bin/python -u -m stage_III.libero.full_train --run /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/libero/runs/grpo_full800_round2_20261013_115427 --initial-actor /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/libero/checkpoints/round_01_full800_20261013/actor_final.pt --gpus 0,2,2,5,5,7,7 --ports 29800,29801,29802,29803,29804,29805,29806 --groups-per-update 32 --tmux-session libero_stage3_round2 2>&1 | tee /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/libero/launchers/round2_20261013_115427/console.log
train_exit=${PIPESTATUS[0]}
date --iso-8601=seconds > /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/libero/launchers/round2_20261013_115427/finished_at.txt
echo "$train_exit" > /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/libero/launchers/round2_20261013_115427/exit_code.txt
echo "[Stage III][第二轮] 训练进程已退出，退出码=$train_exit；日志和 tmux 窗口继续保留。"
exit "$train_exit"
