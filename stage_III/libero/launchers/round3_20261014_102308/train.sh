#!/usr/bin/env bash
set -uo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
export PYTHONUNBUFFERED=1
export LIBERO_HOME=/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2/runtime/LIBERO
export LIBERO_CONFIG_PATH=/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2/runtime/libero_config
export PYTHONPATH=/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change:/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2/runtime/LIBERO
echo '[Stage III][第三轮] 从第二轮最终 Actor 和 Adam 状态续训；GPU2/5/7 各一个模型 worker。'
/home/wangchaoyang/envs/LaWAM/bin/python -u -m stage_III.libero.full_train --run /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/libero/runs/grpo_full800_round3_20261014_102308 --initial-actor /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/libero/checkpoints/round_02_full800_20261014/actor_final.pt --gpus 2,5,7 --ports 29920,29921,29922 --groups-per-update 32 --tmux-session libero_stage3_round3 2>&1 | tee /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/libero/launchers/round3_20261014_102308/console.log
train_exit=${PIPESTATUS[0]}
date --iso-8601=seconds > /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/libero/launchers/round3_20261014_102308/finished_at.txt
echo "$train_exit" > /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/stage_III/libero/launchers/round3_20261014_102308/exit_code.txt
echo "[Stage III][第三轮] 训练进程退出，退出码=$train_exit。"
exit "$train_exit"
