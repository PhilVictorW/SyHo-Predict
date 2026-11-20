#!/usr/bin/env bash
set -euo pipefail
A2_FIXED_WORK="$(cd -- "$(dirname -- "$0")" && pwd -P)"
source "$A2_FIXED_WORK/runtime_env.sh"
exec /home/wangchaoyang/envs/LaWAM/bin/python -B -u -m a2_stage2_fixed.launch "$@"
