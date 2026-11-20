#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"
bash "$SCRIPT_DIR/start_training.sh" --execute >> "$SCRIPT_DIR/logs/train_gpu4.log" 2>&1
result=$?
printf '%s\n' "$result" > "$SCRIPT_DIR/logs/exit_code.txt"
exit "$result"
