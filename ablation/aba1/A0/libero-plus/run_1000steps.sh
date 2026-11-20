#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/ablation/aba1/A0/libero-plus
exec bash train.sh --execute --after-eval "$@"
