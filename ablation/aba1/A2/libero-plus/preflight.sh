#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
source "$SCRIPT_DIR/runtime_env.sh"
export PYTHONDONTWRITEBYTECODE=1
exec /home/wangchaoyang/envs/LaWAM/bin/python -B "$SCRIPT_DIR/preflight.py" "$@"
