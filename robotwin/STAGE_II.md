# RoboTwin Stage II — 2026-09-25

## Initialization and optimization

Frozen posterior teacher and independent student initialized from `runs/stage_I_20260924/step_001000/model.pt`.
Teacher uses real future frames; student encoding uses only current head/left-wrist/right-wrist images and language.
Teacher remains in eval/no_grad and is excluded from AdamW. Teacher storage independence and unchanged parameter versions are checked.
Student inherits Qwen, original queries, frozen vision/DINO, and the full Stage I hierarchical Flow head.
Student Goal/Local queries and Goal-to-Local relevance gates are newly initialized.
RoboTwin interfaces stay at three views, 30 Hz, 8-step horizon, 16 EEF dimensions, Agilex id 1, no state.

Train GPUs: physical 2 and 5. Microbatch 1/rank, accumulation 16/rank, effective batch 32.
10,000 successful student optimizer updates; failed numerical attempts do not advance the step or scheduler.
New FP32-master CPU AdamW on rank zero; LR 1e-5, warmup 400 updates, cosine floor 5%, gradient clipping 1.
Both ranks average gradients; updated student parameters are broadcast to both ranks. Teacher stays fixed at source step 1000.
Four logged losses: action flow, Local distillation, Goal distillation, total (weights 1/1; cosine + Gram at layers 4/8/12/16).
Fixed 64-sample offline validation at 0 and each 250 steps. It is separate from simulator evaluation.

Nonfinite losses/gradients are detected collectively. Both ranks discard the entire accumulated batch without optimizer/scheduler update,
record sample episode/frame IDs and bad parameter names in `anomalies.jsonl`, and continue with new data.
Gradient norms are accumulated in FP32/FP64, not BF16. Ten consecutive invalid updates cause a visible diagnostic failure rather than corrupt weights.
This prevents the previous single nonfinite-gradient event from terminating training; it cannot guarantee immunity to hardware/process failures.

## Saving and storage

Run: `runs/stage_II_20260925`. Every 500 successful updates, save the complete student model and processor.
Keep ALL 500,1000,...,10000 model snapshots. `step_000001` is an additional launch-verification snapshot and is not part of formal quick evaluation.
Keep the newest two full optimizer/RNG/scheduler states. Because HDD had only 162GB free, optimizer files are stored persistently on the system disk:
`/home/wangchaoyang/CKPT/LaWAM/robotwin_stageII_optimizer/stage_II_20260925/step_XXXXXX/trainer.pt`.
Each project checkpoint has a `trainer.pt` symlink. Completion markers are written only after model and optimizer writes complete.
`--resume <project checkpoint>/trainer.pt` restores weights, masters, moments, LR, per-rank RNG, successful/attempt counters and sampler cursor.
Keep the project checkpoint symlink path when resuming (the physical optimizer path alone lacks the student model).

## Asynchronous simulator evaluation

`robotwin.eval_queue` is a separate tmux process. It discovers completed 500-step snapshots in ascending order and retains every checkpoint in the queue.
GPU 0 and GPU 5 each run one policy worker with two simulator slots. GPU 5 is shared with training; evaluation may reduce throughput but never requests training pause.
Each snapshot: all 50 official tasks, demo_clean and demo_randomized, 10 episodes/task/mode, target 1000 episodes.
Action replanning interval is 8, no action ensemble, no videos. Client handles one denormalization, gripper inversion and quaternion normalization.
A task's inactivity timer starts AFTER the policy server has loaded its weights and become ready.
After 2400 seconds without another completed episode, terminate only that task, write `skipped_summary.json`, and claim the next pending task.
Timeouts, startup failures and completed rollouts remain distinguishable; success rates use actual completed episodes, with coverage separately recorded.
Results: `quick_eval/step_XXXXXX_demo_*.json`, `summary.json`, `per_task_results.csv`, per-task progress/logs.
`smoke_eval` is a separate one-episode-per-mode integration test using the first-update student; it is not a benchmark result.

## Monitoring

`tmux attach -t robotwin_stageII_progress`: defaults to a Chinese overview, refreshing every 2 seconds.
Press `1` for overview, `2` for full-history Action/Local/Goal plots, `3` for all twenty 500-step checkpoint evaluations; PgUp/PgDn scroll the overview table.
The overview shows training progress, ETA, latest/50-step means/fixed-validation losses, live rollout progress, and clean/randomized/overall success rates.
Separate tmux windows: dashboard, loss_log, loss_curves, quick_eval. Missing evaluations display `--`, and timeout skips are reported separately.
Charts and CSV are exported every minute by the separate `loss_export` window; UI refresh never waits for matplotlib.
`tmux attach -t lawam_robotwin_stageII`: train, eval_queue, smoke windows/logs.
Complete histories: `stage_II_loss_history.csv`, `stage_II_loss_curve.png`, `stage_II_loss_curve.svg`.

## Shared integration edits

`../goal_local/model.py`: factory uses `type(posterior)` so RoboTwin students retain their adapter.
`../examples/Robotwin/eval_files/batched_eval_runner.py`: server module configurable via ROBOTWIN_SERVER_MODULE; defaults unchanged.
Everything else added for this run lives under `robotwin/`, except optimizer storage above.
Original aba2 evaluation was stopped at user request; original dataset, release weights and Stage I snapshots are preserved.
