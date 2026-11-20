#!/usr/bin/env bash
set -euo pipefail
WORK="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$WORK/../.."
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
PY=/home/wangchaoyang/envs/LaWAM/bin/python
"$PY" -m stage_III.robotwin.check "$@"
"$PY" -m stage_III.robotwin.tests.run_checks
