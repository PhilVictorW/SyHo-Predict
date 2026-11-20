#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
cd "$SCRIPT_DIR"
receipt="$SCRIPT_DIR/logs/stage_II_exit_code.txt"
if [[ -e "$receipt" ]]; then
    echo "Stage II already has an exit receipt; refusing a duplicate launch" >&2
    exit 64
fi
printf '\n[AUTO] Starting adaptive Stage II: one rank each on GPUs 0,2,7 and measured ranks on GPU5 after verified Teacher3000\n' >> "$SCRIPT_DIR/logs/train_gpu6.log"
bash "$SCRIPT_DIR/start_stage_ii_adaptive.sh" --execute >> "$SCRIPT_DIR/logs/train_gpu6.log" 2>&1
result=$?
printf '%s\n' "$result" > "$receipt.$$.tmp"
mv "$receipt.$$.tmp" "$receipt"
exit "$result"
