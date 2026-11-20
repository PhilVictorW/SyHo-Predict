#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
cd "$SCRIPT_DIR"
if [[ $# -eq 2 && "${1:-}" == "--resume" ]]; then
    /home/wangchaoyang/envs/LaWAM/bin/python -B -u "$SCRIPT_DIR/resume_stage_i.py" --resume "$2" --execute >> "$SCRIPT_DIR/logs/train_gpu6.log" 2>&1
elif [[ $# -eq 0 ]]; then
    bash "$SCRIPT_DIR/start_training.sh" --execute >> "$SCRIPT_DIR/logs/train_gpu6.log" 2>&1
else
    echo 'Usage: run_logged.sh [--resume CHECKPOINT]' >&2
    exit 64
fi
result=$?
printf '%s\n' "$result" > "$SCRIPT_DIR/logs/exit_code.txt.tmp"
mv "$SCRIPT_DIR/logs/exit_code.txt.tmp" "$SCRIPT_DIR/logs/exit_code.txt"
exit "$result"
