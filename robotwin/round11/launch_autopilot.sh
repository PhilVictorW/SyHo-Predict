#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD"
mkdir -p robotwin/round11/autopilot
exec /usr/bin/python3 -u -m robotwin.round11.autopilot >>robotwin/round11/autopilot/supervisor.log 2>&1
