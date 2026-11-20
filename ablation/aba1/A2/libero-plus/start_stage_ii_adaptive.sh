#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
source "$SCRIPT_DIR/runtime_env.sh"
exec /home/wangchaoyang/envs/LaWAM/bin/python -B -u -m a2_stage2.launch "$@"
