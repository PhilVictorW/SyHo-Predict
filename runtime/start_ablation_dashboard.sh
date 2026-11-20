#!/usr/bin/env bash
set -euo pipefail
SESSION=lawam_aba2_dashboard
REPO=/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" -x 180 -y 45 -c "$REPO" '/home/wangchaoyang/envs/LaWAM/bin/python -u runtime/watch_ablation_dashboard.py'
fi
printf 'View: tmux attach -t %s\nDetach: Ctrl-b, then d\n' "$SESSION"
