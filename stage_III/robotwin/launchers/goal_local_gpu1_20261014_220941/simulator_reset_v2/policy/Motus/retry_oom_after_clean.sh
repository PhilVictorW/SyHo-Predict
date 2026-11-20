#!/bin/bash
set -euo pipefail

ROBOTWIN_ROOT=/mnt/hdd/wangchaoyang/RoboTwin
RANDOM_ROOT=/mnt/hdd/wangchaoyang/results/Motus
CLEAN_ROOT=/mnt/hdd/wangchaoyang/results/Motus/clean
WATCH_LOG="${RANDOM_ROOT}/oom_retry_after_clean.log"
LOCK_DIR="${RANDOM_ROOT}/oom_retry_after_clean.lock"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "[$(date '+%F %T')] OOM retry watcher already active; exiting" >> "$WATCH_LOG"
    exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

mkdir -p "$RANDOM_ROOT" "$CLEAN_ROOT"

echo "[$(date '+%F %T')] OOM retry watcher started" >> "$WATCH_LOG"
echo "[$(date '+%F %T')] waiting for clean run_state.json to become completed" >> "$WATCH_LOG"

while true; do
    clean_status=$(/usr/local/miniforge3/bin/python - <<'PY_INNER'
import json
from pathlib import Path
p = Path('/mnt/hdd/wangchaoyang/results/Motus/clean/run_state.json')
try:
    print(json.loads(p.read_text()).get('status', ''))
except Exception:
    print('')
PY_INNER
)
    if [ "$clean_status" = "completed" ]; then
        break
    fi
    sleep 300
done

cd "$ROBOTWIN_ROOT"
eval "$(conda shell.bash hook)"
conda activate /home/wangchaoyang/envs/motu_robo

export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export HF_HOME=/mnt/hdd/hf_a100/wangchaoyang
export HF_HUB_CACHE=/mnt/hdd/hf_a100/wangchaoyang/hub

run_oom_retry() {
    local label="$1"
    local root="$2"
    local task_config="$3"
    echo "[$(date '+%F %T')] starting ${label} OOM-only retry" >> "$WATCH_LOG"
    python policy/Motus/run_quantized_eval_resume.py \
        --config-file policy/Motus/paths_config.yml \
        --tasks-file tasks_all.txt \
        --gpus 0,1,2,3,6 \
        --min-free-mem-mib 43000 \
        --stall-timeout-sec 1800 \
        --poll-sec 60 \
        --result-root "$root" \
        --task-config "$task_config" \
        --retry-oom-only \
        >> "$WATCH_LOG" 2>&1
    echo "[$(date '+%F %T')] finished ${label} OOM-only retry" >> "$WATCH_LOG"
}

if [ -d "${RANDOM_ROOT}/randomized_oom_retry_now.lock" ]; then
    echo "[$(date '+%F %T')] immediate randomized OOM retry is active; waiting before watcher randomized pass" >> "$WATCH_LOG"
    while [ -d "${RANDOM_ROOT}/randomized_oom_retry_now.lock" ]; do
        sleep 300
    done
fi

run_oom_retry randomized "$RANDOM_ROOT" demo_randomized_motus
run_oom_retry clean "$CLEAN_ROOT" demo_clean

echo "[$(date '+%F %T')] OOM retry watcher finished" >> "$WATCH_LOG"
