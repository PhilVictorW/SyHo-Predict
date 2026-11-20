#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
exec /home/wangchaoyang/envs/LaWAM/bin/python -B -u stage_III/robotwin/launchers/goal_local_gpu1_20261014_220941/train_direct_gpu3_v2.py \
 --config stage_III/robotwin/launchers/goal_local_gpu1_20261014_220941/workflow_direct_gpu3.json \
 --student stage_III/robotwin/runs/goal_local_resetv2_20261015_1238/interaction/checkpoints/student_000200.pt \
 --source-run stage_III/robotwin/runs/goal_local_resetv2_20261015_1238 \
 --train-manifest stage_III/robotwin/runs/goal_local_resetv2_20261015_1238/manifests/prepared_train.json \
 --run-dir stage_III/robotwin/runs/goal_local_resetv2_20261015_1238/rl \
 --gpu 3 --defer-pretraining-evaluation --execute "$@"
