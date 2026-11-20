#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change
GL_STAGE="${1:?Pass I or II}"
GL_PYTHON=/home/wangchaoyang/envs/LaWAM/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ "$GL_STAGE" == I ]]; then
  export CUDA_VISIBLE_DEVICES=5
  GL_RUN=runs/goal_local/stage_I_resume2000_20260921
  GL_RECOVERY=(--resume runs/goal_local/stage_I_20260921/step_002000/trainer.pt)
  GL_DIAGNOSTICS=8
elif [[ "$GL_STAGE" == II ]]; then
  export CUDA_VISIBLE_DEVICES=0
  GL_RUN=runs/goal_local/stage_II_warm250_20260921
  GL_RECOVERY=(--warm-start runs/goal_local/stage_II_teacher2000_20260921/best
    --stage-one runs/goal_local/teachers/stage_I_step_002000)
  GL_DIAGNOSTICS=0
else
  echo 'Expected I or II' >&2; exit 2
fi
mkdir -p "$GL_RUN"
# Later restarts use the newest complete checkpoint from this recovery run.
GL_LATEST=''
for GL_MARKER in "$GL_RUN"/step_*/checkpoint_complete.json; do
  if [[ -f "$GL_MARKER" && -f "${GL_MARKER%/*}/trainer.pt" ]]; then
    GL_LATEST="${GL_MARKER%/*}/trainer.pt"
  fi
done
if [[ -n "$GL_LATEST" ]]; then
  GL_RECOVERY=(--resume "$GL_LATEST")
  if [[ "$GL_STAGE" == II ]]; then
    GL_RECOVERY+=(--stage-one runs/goal_local/teachers/stage_I_step_002000)
  fi
elif [[ -f "$GL_RUN/metrics.jsonl" ]]; then
  echo 'Existing metrics but no full checkpoint; inspect before restarting.' >&2
  exit 1
fi
trap 'code=$?; printf "Launcher exit code: %s\n" "$code" >> "$GL_RUN/launcher.log"' EXIT
"$GL_PYTHON" -m goal_local.train --config "configs/goal_local/stage_${GL_STAGE}.json" --stage "$GL_STAGE" \
  --output "$GL_RUN" --steps 10000 --batch-size 2 --accumulation 16 \
  --workers 4 --cpu-threads 4 --device cuda:0 --eval-batches 64 --diagnostic-batches "$GL_DIAGNOSTICS" \
  --save-every 250 --eval-every 250 --keep-checkpoints 2 --optimizer-device cpu \
  --save-on-first-update "${GL_RECOVERY[@]}" >> "$GL_RUN/train.log" 2>&1
