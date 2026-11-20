#!/usr/bin/env bash
set -euo pipefail
WORK="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$WORK/../.."
export PYTHONDONTWRITEBYTECODE=1
exec /home/wangchaoyang/envs/LaWAM/bin/python -m stage_III.robotwin.monitor "$@"
