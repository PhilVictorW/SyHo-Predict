# RoboTwin Goal/Local training

Stage II student training and asynchronous 50×2×10 evaluation: see [STAGE_II.md](STAGE_II.md).
The Stage I notes below describe the earlier run.

This experiment reuses `goal_local` role attention, language execution and hierarchical Action DiT.
Adapters live only in `robotwin/`; existing LIBERO training and aba2 evaluation files are unchanged.

## Sources

- Initial model: `/mnt/hdd/wangchaoyang/CKPT/LaWAM/results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt`.
- Statistics: the same release's `dataset_statistics.json`, `agilex` key.
- Data: `/mnt/hdd/wangchaoyang/CKPT/LaWAM/dataset/robotwin_merged`, read-only.
- Load through `robotwin.model.RobotwinPolicy.from_baseline`; strict retained Qwen/DINO loads and whitelist-checked Flow load, inherited from `goal_local.model`.
- New modules are initialized fresh. No LIBERO teacher/student weights are used.

## Data contract

27,500 episodes, 6,077,247 frames, three cameras, 30 Hz. A seeded episode-level random 95/5 split prevents episode overlap (26,125 train / 1,375 validation).
Do not reuse the LIBERO per-instruction split: many RoboTwin instructions are singletons.
The cache contains only action/timestamp arrays and episode metadata; videos remain at the source and are decoded on demand.
RGB views are resized to 256x256 using bilinear interpolation.

Current context uses the main and both wrist cameras. Local uses all three future views; Goal uses five main-view future frames.
The two-view Local view embedding is extended to three; role slots and the two role pathways are unchanged.
H=8 at 30 Hz (0.2667 seconds), Local t+8, Goal t+[24,40,56,72,88]. This retains the Goal/Local 8-action design, not the original RoboTwin policy's 36-action chunk.
No sample crosses an episode boundary. Missing futures mask the corresponding posterior role; tail actions supervise only real steps.

Action packing is left xyz, left wxyz quaternion, left gripper, right xyz, right wxyz quaternion, right gripper (16 dimensions).
Position and gripper min/max come from the inherited checkpoint. Quaternions pass through unchanged; normalized grippers are sign-inverted as in RobotwinEEFDataConfig.
Actions use a 50x32 padded buffer with only up to 8x16 valid values. Agilex embodiment id is 1. No proprioceptive state is used.

## Training and monitoring

`bash robotwin/launch.sh` starts Stage I for 4,000 optimizer updates on physical GPU 5.
Microbatch 1, accumulation 32 (effective batch 32), LR 1e-5, 400-step warmup and cosine decay, BF16 forward, FP32 master AdamW on CPU, gradient clipping 1.0.
Qwen vision/merger and DINO are frozen; language, retained queries, Flow and new future/role modules train.
Only posterior action flow loss trains Stage I. This is not a current-only deployable student.
The PyTorch allocator is capped at 40% of GPU memory to leave space for the ongoing aba2 simulator/policy; sharing GPU 5 reduces throughput.

Run: `robotwin/runs/stage_I_20260924`.
Metrics every optimizer update; fixed 64-batch validation at step 0 and every 250 steps. Eight validation batches also measure flow loss with role injection disabled using identical noise.
Model snapshots at 500,1000,...,4000 are all kept, plus a first-update safety snapshot and the best-validation model. Full optimizer recovery states are kept only for the latest two snapshots to control disk use.
Checkpoint files: `model.pt`, `trainer.pt` (latest two only), processor and completion marker.
Load models using `robotwin.model.RobotwinPolicy.load`, not the LIBERO-only base loader.

`python -m robotwin.monitor --run robotwin/runs/stage_I_20260924` maintains `loss_history.csv`, `loss_summary.json`, `loss_curve.png`, `loss_curve.svg`, and `REPORT.md`.
Loss is an offline metric; its decline does not establish online RoboTwin success rate.

Resume from an available complete checkpoint using the launch arguments plus `--resume <checkpoint>/trainer.pt`; this inherited trainer restores optimizer/RNG but starts a new data iterator, so does not promise sample-exact continuation.

Live tmux session: `tmux attach -t lawam_goal_local_robotwin_stageI` (`progress` window).
Preflight passed: `verification/gpu.json`; strict inherited weights, 16 decoded boundary samples, spawn data workers, true nonzero role/language/query gradients, finite all-missing-future tail backward.

## AV1 source integrity finding

The official `videos/observation.images.cam_high/chunk-000/file-082.mp4` contains two undecodable frames near 29.70/29.73 seconds with both libdav1d and libaom. Its local SHA-256 matches the Hugging Face LFS hash; this is not an incomplete download.
The reader recovers past invalid packets but requires every requested frame within 18 ms of its timestamp. Unreadable requested frames reject that anchor; up to 32 deterministic neighboring anchors within the same episode are tried and all replacements are logged in `verification/decode_events.<pid>.jsonl`. No wrong-time or black image is used to replace a valid requested frame. The initial failed validation attempt is archived inside the run.
All 64 fixed validation anchors and a deliberate corrupt-target resampling case are checked by `verification/decoder_recovery.json`.
