#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
export CUDA_VISIBLE_DEVICES=2,5 CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec /home/wangchaoyang/envs/LaWAM/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    --module robotwin.train_stage_two --stage-one robotwin/runs/stage_I_20260924/step_001000 \
    --output "$1" --optimizer-dir /home/wangchaoyang/CKPT/LaWAM/robotwin_stageII_optimizer/stage_II_20260925 --steps 10000 --accumulation 16 --workers 2 --cpu-threads 4 \
    --save-every 500 --eval-every 250 --eval-batches 64 --warmup 400 --gpu-memory-fraction .36 "${@:2}"
