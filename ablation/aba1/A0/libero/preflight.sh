#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONDONTWRITEBYTECODE=1
exec /home/wangchaoyang/envs/LaWAM/bin/python preflight.py "$@"
