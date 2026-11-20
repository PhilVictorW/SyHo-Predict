#!/bin/bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec nice -n 10 ionice -c 2 -n 7 /home/wangchaoyang/envs/LaWAM/bin/python -m robotwin.round10.candidate_queue >> robotwin/round10/candidate_queue.log 2>&1
