#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD"
mkdir -p robotwin/round11/adapter_workflow
exec /usr/bin/python3 -u -m robotwin.round11.adapter_workflow >>robotwin/round11/adapter_workflow/supervisor.log 2>&1
