# GPU 2,5 Stage I continuation

Launched command on 2026-09-24 after user approval to resume from step 1000:

```bash
bash robotwin/launch_distributed.sh \
  robotwin/runs/stage_I_20260924/step_001000/trainer.pt \
  robotwin/runs/stage_I_ddp_20260924
```

- Two DDP ranks, each batch 1 with 16 accumulation microsteps: global batch remains 32.
- The posterior model, FP32 AdamW master weights/moments, scheduler and global step are restored; total target remains 4000 updates.
- Gradients are averaged by DDP. Rank zero alone owns the CPU optimizer and broadcasts updated weights to rank one.
- Same fixed 64 validation samples, fixed validation noise, 250-step validation and 500-step checkpoints.
- A fresh output directory copies original metrics through the selected recovery step. The complete original run is preserved separately.
- DistributedSampler shards frames without overlap. Transition from the old single-GPU sampler changes sample order; it is not sample-for-sample replay.
- The first resumed update also saves a full recovery checkpoint. All 500-step model snapshots are retained; only two latest optimizer snapshots are kept in the new run.
- CPU two-process numerical verification passed, including conditional unused gradients, accumulation and optimizer restoration. GPU2 real-model preflight is recorded separately when completed. GPU5 currently refuses new CUDA primary contexts despite free VRAM.
- Original GPU5 trainer was stopped after completing step 1039; its entire run remains untouched except its status marking the migration. Unsaved parameter updates after step 1000 are intentionally abandoned with user approval.
- Releasing the original trainer allowed a fresh GPU5 CUDA context to be created. Both new ranks initialized NCCL successfully; actual model restore, validation and first resumed update are checked before declaring migration complete.
- Existing tmux dashboard robotwin_stageI_progress now follows the new output directory. Training runs in lawam_goal_local_robotwin_stageI:train_gpu25.

Verified: both GPU ranks completed global optimizer step 1001. Action flow loss 0.0046873996; LR 9.3615465e-6. Restored validation flow 0.0029135223 versus original 0.0029105694 (~0.10% relative difference). First extra checkpoint is being written; consult checkpoint_complete.json for completion.
