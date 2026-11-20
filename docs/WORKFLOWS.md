# Training and inference

Commands below describe the entry points in this source release. Run them **from a newly configured clone**, with external data, baseline weights and simulator assets prepared as described in [ASSETS.md](ASSETS.md). These commands were not executed during publication.

Activate the model environment for training and policy services. The rollout orchestrators launch their own simulator interpreters. Choose available GPUs explicitly; historical shell scripts and evaluation queues retain the original GPU assignments, dates, budgets and resume conditions.

## Stage overview

| Stage | Input | Trainable component | Output |
| --- | --- | --- | --- |
| I | Current observations, instructions, future demonstrations | Posterior teacher and action policy | Teacher checkpoint |
| II | Current observations and teacher supervision | Prior student; teacher frozen | Student checkpoint |
| III | Student rollouts and environment rewards | Continuous gating actor; student frozen | Gating actor checkpoint |

Future observations supervise training. Deployed student inference uses current images and language. Stage I posterior policies are evaluated with offline losses/diagnostics; closed-loop deployment uses Stage II students or Stage III students with the learned actor.

## LIBERO

### Stage I and II

```bash
# Stage I: example run through the selected teacher step.
CUDA_VISIBLE_DEVICES=0 python -m goal_local.train \
  --stage I --config configs/goal_local/stage_I.json \
  --output runs/goal_local/stage_I_new --steps 2000 \
  --batch-size 1 --accumulation 32 --save-every 1000

# Stage II: an independent student initialized from the posterior teacher.
CUDA_VISIBLE_DEVICES=0 python -m goal_local.train \
  --stage II --config configs/goal_local/stage_I.json \
  --stage-one runs/goal_local/stage_I_new/step_002000 \
  --output runs/goal_local/stage_II_new --steps 4000 \
  --batch-size 1 --accumulation 32 --save-every 500
```

These are fresh-run examples, not exact replay commands for the original warm-start trajectory. The historical selection used teacher step 2000 and student step 4000; its arguments and warm-start provenance are in the checkpoint archive.

### Student inference and evaluation

For action prediction from two images:

```bash
CUDA_VISIBLE_DEVICES=0 python -m goal_local.predict \
  --checkpoint runs/goal_local/stage_II_new/step_004000 \
  --main /path/to/main.png --wrist /path/to/wrist.png \
  --instruction "put the bowl on the plate" --output actions.json
```

The output is an 8-step action chunk at 20 Hz. For simulator evaluation, the implementation is split across:

- `goal_local/eval_archive.py`: retain completed evaluation snapshots.
- `goal_local/eval_server.py`: model service, accepting archived student snapshots with completion markers.
- `goal_local/eval_task.py`: one LIBERO task in the simulator environment.
- `goal_local/eval_queue.py`: multi-checkpoint 40-task evaluation.
- `goal_local/eval_full.py`: historical student-4000 evaluation, 40 tasks × 50 episodes.

The queue uses `TRAIN`, `SNAPSHOTS` and `OUTPUT` from `goal_local/eval_common.py`. Point these at your run, prepare the expected snapshots, and choose GPUs before launching. The full evaluator has a fixed historical checkpoint and GPU layout; it is not a generic arbitrary-checkpoint CLI.

### Stage III

The original trainer verifies the frozen teacher/student and baseline evaluation against `stage_III/libero/baseline_snapshot.json`. Supply those matching archived files before using this historical recipe:

```bash
python -m stage_III.libero.train \
  --gpu 0 --port 29650 \
  --iterations 2 --groups-per-iteration 4 --trajectories-per-group 4 \
  --run stage_III/libero/runs/pilot_new
```

`full_train.py` contains the longer multi-worker recipe. The policy service is `server.py`; simulator rollouts use `rollout.py`. Final evaluation uses:

```bash
python -m stage_III.libero.eval_full --manifest /path/to/eval_manifest.json --round 1
```

The final-evaluation manifest is experiment-specific; inspect `stage_III/libero/eval_common.py` and the historical launchers for its schema. New Stage II students require a new baseline evaluation and corresponding identity/score checks; changing only a checkpoint path is insufficient.

## LIBERO-Plus

### Stage I and II

Both trainers run from `libero-plus/`. Stage I requires two ranks. Stage II intentionally uses **four ranks on two physical GPUs**, with CPU Gloo synchronization and two ranks mapped to each GPU.

```bash
cd libero-plus

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 runtime/train_ddp.py \
  --config configs/stage_I.json --output runs/stage_I_new \
  --steps 5000 --batch-size 2 --accumulation 8 --save-every 1000

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 runtime/train_stage_two.py \
  --teacher runs/stage_I_new/step_005000 \
  --output runs/stage_II_new --optimizer-dir runs/stage_II_new/optimizer \
  --steps 4000 --batch-size 1 --accumulation 8 --save-every 500

cd ..
```

The selected archived teacher is step 5000 and student is step 4000. Keep the teacher frozen. The indexed RLDS dataset location and source LaWAM LIBERO checkpoint are specified in `libero-plus/configs/stage_I.json`.

### Student inference and evaluation

The same `goal_local.predict` entry point can produce raw student actions from main/wrist images. For benchmark scoring use the Plus-specific client and action protocol:

- `libero-plus/runtime/student_server.py`: current-observation model service.
- `student_rollout.py` and `student_action_protocol.py`: simulator rollout and gripper/action conventions.
- `student_eval_queue.py`: shared evaluator.
- `student_eval_steps2500_4000.py`: archived four-checkpoint evaluation schedule.

Update `student_eval_common.py` to select your training directory and simulator. The historical schedule requires its stratified task manifest and fixes checkpoint/GPU pairs; raw predicted actions alone are not a complete simulator protocol.

### Stage III

The archived baseline identity is in `stage_III/libero_plus/baseline_snapshot.json`. With matching student, action semantics and baseline results restored:

```bash
python -m stage_III.libero_plus.train \
  --gpu 0 --port 29650 \
  --iterations 2 --groups-per-iteration 4 --trajectories-per-group 4 \
  --run stage_III/libero_plus/runs/pilot_new
```

`full_train.py` implements the extended recipe. After a completed compatible training run:

```bash
python -m stage_III.libero_plus.evaluate_after_train \
  --train-run stage_III/libero_plus/runs/your_completed_run \
  --folder stage_III/libero_plus/evaluation/your_evaluation \
  --prepare-only
```

Inspect the prepared evaluation before removing `--prepare-only`. Supply the benchmark's `LIBERO_CONFIG_PATH/config.yaml` with valid benchmark, asset, BDDL and initial-state paths; the original external simulator configuration is not bundled.

## RoboTwin 2.0

### Original Stage I / II branch

```bash
CUDA_VISIBLE_DEVICES=0 python -m robotwin.train \
  --stage I --config robotwin/configs/stage_I.json \
  --output robotwin/runs/stage_I_new --steps 1000 --save-every 1000

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module robotwin.train_stage_two \
  --stage-one robotwin/runs/stage_I_new/step_001000 \
  --output robotwin/runs/stage_II_new \
  --optimizer-dir robotwin/runs/stage_II_new/optimizer \
  --steps 3000 --accumulation 16 --save-every 500
```

Stage II requires two ranks, a step-1000 posterior teacher, and global batch 32. See [the original Stage II notes](../robotwin/STAGE_II.md). This branch uses three camera views and 16-dimensional dual-arm actions.

`robotwin/eval_server.py` serves the policy; `examples/Robotwin/eval_files/batched_eval_runner.py` launches the simulator workers. The historical queue is:

```bash
python -m robotwin.eval_queue --run robotwin/runs/stage_II_new
```

Review the fixed GPU assignments in `robotwin/eval_queue.py` first. It evaluates clean/randomized tasks and applies the required denormalization, gripper and quaternion conventions. Its completion condition reflects the original 10,000-step schedule, so a shorter run may leave it waiting for more checkpoints.

### Final interaction branch and Stage III

The final Stage III branch has a different lineage:

```text
LaWAM RoboTwin SFT base
  -> round4 future_aux adapter (step 200)
  -> round6 control adapter (step 200)
  -> round7 query/gate initializers
  -> interaction student (step 200)
  -> continuous gating actor
```

The earlier teacher-1000/student-3000 branch is retained for provenance and must not be substituted for this interaction student. The supporting training code is retained in `robotwin/round4/`, `round6/`, `round7/` and later rounds.

With the compatible base/adapters supplied, the workflow connects interaction distillation, sensitivity checks, reproducible scene preparation, paired validation, admission, gating optimization and final evaluation:

```bash
# Print the workflow plan.
python -m stage_III.robotwin.workflow \
  --config stage_III/robotwin/configs/workflow.json \
  --run-dir stage_III/robotwin/runs/new_experiment --gpu 0

# Execute a configured workflow, including held-out final evaluation.
python -m stage_III.robotwin.workflow \
  --config stage_III/robotwin/configs/workflow.json \
  --run-dir stage_III/robotwin/runs/new_experiment --gpu 0 \
  --execute --final-evaluation
```

The default is a small pilot, not the full 50-task benchmark. For evaluation of a compatible student and actor:

```bash
python -m stage_III.robotwin.evaluate \
  --config stage_III/robotwin/configs/workflow.json \
  --student /path/to/student_000200.pt --actor /path/to/actor_0100.pt \
  --manifest /path/to/prepared_test.json \
  --output stage_III/robotwin/evaluation/new_evaluation --gpu 0 --execute
```

See [the detailed RoboTwin workflow](../stage_III/robotwin/README.md) for the state/action interface and admission rules. That document also contains dated development notes; they describe the original development state.

Archived RoboTwin checkpoints check configuration equality, file paths, sizes, mtimes and source hashes. Moving an archived run to a new machine does not satisfy these identities automatically. Preserve the checks and establish a matching relocated configuration/baseline or generate a new run; do not bypass them to claim an equivalent resumed experiment.

## Scope of validation

Publication checks parse Python, shell and JSON source and exercise the new path configuration helper on a disposable copy. No model training, GPU inference, simulator rollout, environment installation or benchmark rerun is performed as part of this source release. Read [ASSETS.md](ASSETS.md) before attempting a complete run.
