#!/usr/bin/env bash
set -euo pipefail
source /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2/runtime/ablation_env.sh
RUN_DIR="$ABA_ROOT/results/ablation/inference_without_lawm_and_queries/full_40x50_20260915"
mkdir -p "$RUN_DIR/queue"
exec "$STAR_VLA_PYTHON" -u runtime/run_task_queue.py --run-dir "$RUN_DIR" --gpus 0 1 2 3 6 7 >> "$RUN_DIR/queue/launcher.log" 2>&1
