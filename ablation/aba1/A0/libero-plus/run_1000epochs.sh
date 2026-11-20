#!/usr/bin/env bash
# Historical command name retained; the current authorized budget is 1000 updates.
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/ablation/aba1/A0/libero-plus
exec bash run_1000steps.sh "$@"
