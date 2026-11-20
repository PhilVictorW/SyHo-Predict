#!/bin/bash
set -euo pipefail

echo "Starting Motus quantized evaluation on RoboTwin at $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$SCRIPT_DIR"
CONFIG_FILE="${POLICY_DIR}/paths_config.yml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

read_config_value() {
    local key="$1"
    grep "^${key}:" "$CONFIG_FILE" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs
}

ROBOTWIN_ROOT="$(read_config_value robotwin_root)"
CONDA_ENV="$(read_config_value conda_env)"

if [ -z "$ROBOTWIN_ROOT" ] || [ -z "$CONDA_ENV" ]; then
    echo "Error: robotwin_root or conda_env is missing in $CONFIG_FILE"
    exit 1
fi

cd "$ROBOTWIN_ROOT"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found."
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export HF_HOME=/mnt/hdd/hf_a100/wangchaoyang
export HF_HUB_CACHE=/mnt/hdd/hf_a100/wangchaoyang/hub

python policy/Motus/run_quantized_eval.py \
    --config-file policy/Motus/paths_config.yml \
    --tasks-file tasks_all.txt \
    --gpus 1,2,3,6 \
    --min-free-mem-mib 43000 \
    --stall-timeout-sec 1800 \
    --poll-sec 60
