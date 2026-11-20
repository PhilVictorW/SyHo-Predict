#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
if [[ $# -ne 2 ]]; then
    echo 'Usage: bash robotwin/launch_distributed.sh FULL_TRAINER_CHECKPOINT NEW_OUTPUT_DIRECTORY' >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES=2,5 CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec /home/wangchaoyang/envs/LaWAM/bin/python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=2 --module robotwin.train_distributed \
    --resume "$1" --output "$2" --steps 4000 --batch-size 1 --accumulation 16 \
    --workers 2 --cpu-threads 4 --save-every 500 --eval-every 250 \
    --eval-batches 64 --diagnostic-batches 8 --gpu-memory-fraction 0.30 --save-first-update
