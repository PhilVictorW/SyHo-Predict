#!/usr/bin/env bash
set -euo pipefail
source /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2/libero-plus/env.sh
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/libero-plus
export PYTHONPATH="$PWD/runtime:$ABA_BASE:$PWD/..:$LIBERO_HOME:$ABA_PLUS_ROOT/runtime"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec /home/wangchaoyang/envs/LaWAM/bin/python -u runtime/student_eval_steps2500_4000.py
