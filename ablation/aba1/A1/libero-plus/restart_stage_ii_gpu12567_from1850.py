#!/usr/bin/env python3
"""Resume A1 Stage II step 1850 on one rank per GPU 1,2,5,6,7."""
import fcntl, json, os, shutil, subprocess, sys, time
from pathlib import Path

WORK = Path(__file__).resolve().parent
RUN = WORK / "protocol_plus_v1/seed_2026"
OUT = RUN / "II"
LOGS = WORK / "logs"
GPUS = [1, 2, 5, 6, 7]
MAPPING = GPUS.copy()
RESUME = OUT / "step_001850"
PYTHON = "/home/wangchaoyang/envs/LaWAM/bin/python"

sys.path[:0] = [str(WORK), str(WORK.parents[3])]
from a1_local.config_runtime import load_config, atomic_json, checkpoint_path
from a1_local.preflight import check_complete, check_teacher
def training_environment(cfg, devices, cache_dir=None):
    env = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "A1_WORLD_SIZE"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=str(devices), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONPATH=os.pathsep.join([str(WORK), str(WORK.parents[3]), env.get("PYTHONPATH", "")]))
    if cache_dir is not None:
        cache = Path(cache_dir)
        for name in ("matplotlib", "numba", "cuda", "triton", "inductor"):
            (cache / name).mkdir(parents=True, exist_ok=True)
        env.update(XDG_CACHE_HOME=str(cache), MPLCONFIGDIR=str(cache / "matplotlib"),
                   NUMBA_CACHE_DIR=str(cache / "numba"), CUDA_CACHE_PATH=str(cache / "cuda"),
                   TRITON_CACHE_DIR=str(cache / "triton"), TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"))
    return env


def live_training(pid):
    try:
        root = Path("/proc") / str(pid)
        state = (root / "stat").read_text().split(") ", 1)[1][0]
        cmd = (root / "cmdline").read_bytes()
        return state not in ("Z", "X") and b"a1_local.training" in cmd and b"--worker" in cmd
    except (OSError, ValueError):
        return False


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def write_state(phase, **extra):
    row = dict(time=time.time(), pid=os.getpid(), phase=phase,
               physical_gpus=GPUS, rank_to_gpu=MAPPING, training_ranks=len(MAPPING),
               stage="II", resume_step=1850, stage_II_steps=2500,
               checkpoint_policy="every 50 steps, retain latest complete only", **extra)
    atomic_json(LOGS / "auto_continuation_status.json", row)
    atomic_json(LOGS / "gpu_scheduler_status.json", row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def archive_stale_runtime():
    archive = LOGS / ("restart_gpu12567_from1850_" + time.strftime("%Y%m%d_%H%M%S"))
    archive.mkdir(parents=True, exist_ok=False)
    names = ["stage_II_exit_code.txt", "auto_continuation_status.json",
             "auto_continuation_launch.json", "gpu_scheduler_status.json",
             "launch_record.json", "nohup_training.pid", "auto_continuation.pid"]
    names += [p.name for p in OUT.glob("status_rank*.json")]
    names += [p.name for p in OUT.glob("memory_rank*.json")]
    names += [p.name for p in OUT.glob("model_load_rank*.json")]
    names += ["status.json", "sync_audit.json"]
    for name in dict.fromkeys(names):
        candidates = [LOGS / name, OUT / name]
        for path in candidates:
            if path.exists():
                shutil.move(str(path), str(archive / path.name))
                break
    return archive


def main():
    if not RESUME.is_dir():
        raise RuntimeError(f"Missing resume checkpoint: {RESUME}")
    cfg = load_config(WORK / "config.json", "A1")
    meta, marker = check_complete(RESUME, require_resume=True, verify_hash=False)
    if meta.get("step") != 1850 or meta.get("extra", {}).get("stage") != "II":
        raise RuntimeError(f"Unexpected resume checkpoint metadata: {meta}")
    teacher = checkpoint_path(cfg, "I", 3000)
    check_teacher(teacher, cfg)
    for p in OUT.glob("status_rank*.json"):
        row = read_json(p)
        if live_training(row.get("pid", 0)):
            raise RuntimeError(f"Existing A1 worker is still alive: {row.get('pid')}")
    locks = []
    try:
        for path in (LOGS / ".auto_continuation.lock", RUN / ".pipeline.lock"):
            handle = path.open("a")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(handle)
        archive = archive_stale_runtime()
        restart = dict(time=time.time(), archive=str(archive), resume=str(RESUME),
                       resume_step=1850, physical_gpus=GPUS, rank_to_gpu=MAPPING,
                       world_size=5, effective_batch=32, save_every=50,
                       keep_checkpoints=1, reason="A1 previous Stage II OOM at step 1850",
                       previous_exit_code=1, oom_gpu_log="GPU3 reported in PyTorch; physical mapping included GPU7",
                       teacher_step=3000)
        atomic_json(LOGS / "stage_II_gpu12567_restart.json", restart)
        atomic_json(RUN / "pipeline_status.json", dict(phase="training", stage="II", time=time.time(),
                    rank_to_gpu=MAPPING, world_size=5, resume_step=1850,
                    restart_record=str(LOGS / "stage_II_gpu12567_restart.json")))
        env = training_environment(cfg, ",".join(map(str, GPUS)), RUN / "runtime_cache")
        env.update(PYTHONPATH=os.pathsep.join([str(WORK), str(WORK.parents[3]), env.get("PYTHONPATH", "")]),
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TORCH_GLOO_LAZY_INIT="1",
                   MUJOCO_EGL_DEVICE_ID="0", TMPDIR="/tmp")
        command = [PYTHON, "-B", "-u", "-m", "a1_local.training",
                   "--config", str(WORK / "config.json"), "--stage", "II",
                   "--gpus", ",".join(map(str, GPUS)),
                   "--rank-to-gpu", ",".join(map(str, MAPPING)),
                   "--elastic-runtime", "--workers", "1", "--cpu-threads", "2",
                   "--offload-saved-tensors", "--share-frozen-encoders", "--keep-models", "1",
                   "--minimum-free-mib", "8704", "--execute", "--resume", str(RESUME)]
        log_path = LOGS / ("train_stageII_gpu12567_from1850_" + time.strftime("%Y%m%d_%H%M%S") + ".log")
        env["A1_RESTART_RECORD"] = str(LOGS / "stage_II_gpu12567_restart.json")
        with log_path.open("a") as log, open(os.devnull, "rb") as stdin:
            child = subprocess.Popen(command, cwd=WORK, env=env, stdin=stdin,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, pass_fds=tuple(h.fileno() for h in locks))
        atomic_json(LOGS / "launch_record.json", dict(time=time.time(), stage="II",
                    physical_gpus=GPUS, world_size=5, rank_to_gpu=MAPPING,
                    launcher_pid=child.pid, launcher_sid=child.pid, launch_method="nohup",
                    resume_step=1850, stage_II_steps=2500, save_every=50,
                    keep_checkpoints=1, training_log=str(log_path), command=command))
        (LOGS / "auto_continuation.pid").write_text(str(os.getpid()) + "\n")
        (LOGS / "nohup_training.pid").write_text(str(child.pid) + "\n")
        write_state("stage_II", child_pid=child.pid, training_log=str(log_path),
                    archive=str(archive), command=command)
        while child.poll() is None:
            states = [read_json(OUT / f"status_rank{rank}.json") for rank in range(5)]
            steps = [int(x.get("step", 0)) for x in states]
            write_state("stage_II", child_pid=child.pid, stage_II_step=max(steps + [1850]),
                        training_pids=[x.get("pid") for x in states if x.get("pid")],
                        worker_phases=[x.get("phase") for x in states], training_log=str(log_path))
            time.sleep(30)
        code = child.returncode
        (LOGS / "stage_II_exit_code.txt").write_text(str(code) + "\n")
        if code != 0:
            atomic_json(RUN / "pipeline_status.json", dict(phase="failed", stage="II", time=time.time(),
                        exit_code=code, rank_to_gpu=MAPPING, resume_step=1850,
                        training_log=str(log_path), restart_record=str(LOGS / "stage_II_gpu12567_restart.json")))
            write_state("failed", exit_code=code, training_log=str(log_path))
            raise RuntimeError(f"A1 Stage II exited {code}; inspect {log_path}")
        final = OUT / "step_002500"
        check_complete(final, require_resume=True, verify_hash=True)
        atomic_json(RUN / "pipeline_status.json", dict(phase="completed", completed_stages=["I", "II"],
                    teacher_step=3000, student_step=2500, time=time.time(), rank_to_gpu=MAPPING))
        write_state("starting_full_evaluation", student_step=2500)
        eval_env = os.environ.copy()
        eval_env["A1_EVALUATION_GPUS"] = ",".join(map(str, GPUS))
        eval_cmd = [PYTHON, "-B", "-u", str(WORK / "after_training_full_eval.py"),
                    "--execute", "--workers", "5", "--poll-seconds", "10",
                    "--training-exit-file", str(LOGS / "stage_II_exit_code.txt")]
        eval_log = LOGS / "full_eval_gpu12567_from2500.log"
        with eval_log.open("a") as log, open(os.devnull, "rb") as stdin:
            ev = subprocess.Popen(eval_cmd, cwd=WORK, env=eval_env, stdin=stdin,
                                  stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True, pass_fds=(locks[0].fileno(),))
        write_state("full_evaluation", eval_pid=ev.pid, evaluation_gpus=GPUS, evaluation_tasks=10030)
        code = ev.wait()
        if code != 0:
            write_state("failed", eval_exit_code=code, evaluation_log=str(eval_log))
            raise RuntimeError(f"A1 full evaluation exited {code}; inspect {eval_log}")
        result = read_json(RUN / "evaluation" / "full_II_step_002500" / "results.json")
        if not result.get("complete") or result.get("completed") != 10030:
            write_state("failed", evaluation_result=result, evaluation_log=str(eval_log))
            raise RuntimeError(f"A1 full evaluation incomplete: {result}")
        write_state("completed", completed=10030, successes=result.get("successes"),
                    success_rate=result.get("success_rate"))
    finally:
        for handle in reversed(locks):
            try: handle.close()
            except Exception: pass


if __name__ == "__main__":
    main()
