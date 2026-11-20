#!/bin/bash
set -euo pipefail

ROBOTWIN_ROOT=/mnt/hdd/wangchaoyang/RoboTwin
RANDOM_ROOT=/mnt/hdd/wangchaoyang/results/Motus
CLEAN_ROOT=/mnt/hdd/wangchaoyang/results/Motus/clean
WATCH_LOG="${RANDOM_ROOT}/clean_after_randomized_watcher.log"

mkdir -p "$CLEAN_ROOT"

echo "[$(date '+%F %T')] clean watcher started" >> "$WATCH_LOG"
echo "[$(date '+%F %T')] waiting for randomized run_state.json to become completed" >> "$WATCH_LOG"

while true; do
    if [ -f "${CLEAN_ROOT}/run_state.json" ]; then
        clean_status=$(/usr/local/miniforge3/bin/python - <<'PY_INNER'
import json
from pathlib import Path
p=Path('/mnt/hdd/wangchaoyang/results/Motus/clean/run_state.json')
try:
    print(json.loads(p.read_text()).get('status',''))
except Exception:
    print('')
PY_INNER
)
        if [ "$clean_status" = "running" ] || [ "$clean_status" = "completed" ]; then
            echo "[$(date '+%F %T')] clean run already ${clean_status}; watcher exits" >> "$WATCH_LOG"
            exit 0
        fi
    fi

    randomized_status=$(/usr/local/miniforge3/bin/python - <<'PY_INNER'
import json
from pathlib import Path
p=Path('/mnt/hdd/wangchaoyang/results/Motus/run_state.json')
try:
    print(json.loads(p.read_text()).get('status',''))
except Exception:
    print('')
PY_INNER
)
    if [ "$randomized_status" = "completed" ]; then
        break
    fi
    sleep 300
done

echo "[$(date '+%F %T')] randomized completed; launching clean evaluation" >> "$WATCH_LOG"

cd "$ROBOTWIN_ROOT"
eval "$(conda shell.bash hook)"
conda activate /home/wangchaoyang/envs/motu_robo

export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export HF_HOME=/mnt/hdd/hf_a100/wangchaoyang
export HF_HUB_CACHE=/mnt/hdd/hf_a100/wangchaoyang/hub

python policy/Motus/run_quantized_eval_resume.py \
    --config-file policy/Motus/paths_config.yml \
    --tasks-file tasks_all.txt \
    --gpus 0,1,2,3,6 \
    --min-free-mem-mib 43000 \
    --stall-timeout-sec 1800 \
    --poll-sec 60 \
    --result-root "$CLEAN_ROOT" \
    --task-config demo_clean \
    >> "$WATCH_LOG" 2>&1

echo "[$(date '+%F %T')] clean evaluation finished" >> "$WATCH_LOG"
