#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"
receipt="$SCRIPT_DIR/logs/stage_II_exit_code.txt"
if [[ -e "$receipt" ]]; then
    echo "Stage II already has an exit receipt; refusing a duplicate launch" >&2
    exit 64
fi
printf '\n[AUTO] Starting Stage II on GPU5 after verified Teacher3000\n' >> "$SCRIPT_DIR/logs/train_gpu5.log"
bash "$SCRIPT_DIR/train.sh" --stage II --gpus 5 --execute >> "$SCRIPT_DIR/logs/train_gpu5.log" 2>&1
result=$?
printf '%s\n' "$result" > "$receipt.$$.tmp"
mv "$receipt.$$.tmp" "$receipt"
exit "$result"
