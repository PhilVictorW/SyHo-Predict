#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec /home/wangchaoyang/envs/LaWAM/bin/python -B "$SCRIPT_DIR/launch.py" "$@"
