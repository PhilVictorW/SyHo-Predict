# Round10 candidate-only screening with historical baseline

Activated 2026-10-08 by user approval to reuse the existing original LaWAM results. The earlier paired queue is superseded. Its artifacts remain in `online/`; it no longer schedules original evaluations.

## Model and scope

Evaluate the same compatible candidate previously used by Round10: Round7 fixed_queries initialization, Round4 future_aux step 200 adapter, Round6 control step 200 adapter, frozen zero relevance-gate scales. Retains original LaWM conditioning and original queries. Does not use Round8 training updates and does not restart Stage II 3000. This is not proof that the original pure aba2 architecture has been recovered.

50 tasks × clean/randomized × 10 episodes = 1000 candidate episodes. GPU0 and GPU3 each serve the compatible candidate and run at most one simulator slot. The in-flight GPU3 blocks_ranking_rgb clean task was adopted without restart; original GPU0 rollout was stopped with partial artifacts preserved.

## Historical reference

`/mnt/hdd/wangchaoyang/results/LaWAM/RoboTwin/lawam_robotwin_sft_release__{mode}/full_100x50_20260903/tasks/` contains 46 clean and 45 randomized completed tasks, 50 episodes each. Clean: 2126/2300 (92.43%). Randomized: 2017/2250 (89.64%). Missing task-mode baselines remain explicitly unavailable.

For newly scheduled jobs with a baseline, select the first 10 episode records sorted by episode_id, independently of success, and reuse their scene seeds. The adopted job keeps its already selected seeds. Store comparison on overlapping seeds when available. Historical summaries lack instruction text and matched flow-noise records; this is a historical comparison, NOT an exact paired experiment. Candidate 10-episode screening rates and historical 50-episode rates have different sample sizes. Incomplete/error jobs are not counted as zero-success completed evaluations.

Retain current bridge preprocessing, action postprocessing, one simulator slot, 30 Hz, replan 36, no action ensemble, and no future image input. Historical evaluations used four slots. New candidate jobs use the existing deterministic per-query noise procedure, whose equality to historical noise is not established.

## Runtime

Read the actual nested progress.json and summary.json paths. Timeout only after 1800 seconds without bridge-log, model-request, or episode-progress file activity. A long episode with step updates must not be skipped. GPU admission thresholds: 20000 MiB free before starting a server and 10000 MiB free before starting a simulator. These are conservative admission checks, not a reservation against other jobs allocating later. Preserve logs on OOM, cool down 300 seconds, allow at most two attempts per task-mode, preferring another GPU for retries. Other workloads are not stopped or modified.

Queue: `tmux attach -t robotwin_round10_candidate_eval` (logs redirected to candidate_queue.log).
Dashboard: `tmux attach -t robotwin_round10_progress`.
Current outputs: `candidate_online/status.json`, `manifest.json`, `historical_baseline.json`, `jobs/`.
Migration provenance: `candidate_online/migration.json` and `previous_queue_snapshot.json`.
