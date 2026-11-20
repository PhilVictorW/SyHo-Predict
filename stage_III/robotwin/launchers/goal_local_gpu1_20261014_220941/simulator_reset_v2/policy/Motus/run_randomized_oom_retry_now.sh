#!/bin/bash
set -euo pipefail

ROBOTWIN_ROOT=/mnt/hdd/wangchaoyang/RoboTwin
RANDOM_ROOT=/mnt/hdd/wangchaoyang/results/Motus
LOG_FILE="${RANDOM_ROOT}/randomized_oom_retry_now.log"
LOCK_DIR="${RANDOM_ROOT}/randomized_oom_retry_now.lock"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "[$(date '+%F %T')] randomized OOM retry already active; exiting" >> "$LOG_FILE"
    exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

cd "$ROBOTWIN_ROOT"
eval "$(conda shell.bash hook)"
conda activate /home/wangchaoyang/envs/motu_robo

export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export HF_HOME=/mnt/hdd/hf_a100/wangchaoyang
export HF_HUB_CACHE=/mnt/hdd/hf_a100/wangchaoyang/hub

echo "[$(date '+%F %T')] starting immediate randomized OOM-only retry" >> "$LOG_FILE"
python policy/Motus/run_quantized_eval_resume.py \
    --config-file policy/Motus/paths_config.yml \
    --tasks-file tasks_all.txt \
    --gpus 0,1,2,3,6 \
    --min-free-mem-mib 43000 \
    --stall-timeout-sec 1800 \
    --poll-sec 60 \
    --result-root "$RANDOM_ROOT" \
    --task-config demo_randomized_motus \
    --retry-oom-only \
    >> "$LOG_FILE" 2>&1

echo "[$(date '+%F %T')] immediate randomized OOM-only retry finished" >> "$LOG_FILE"
