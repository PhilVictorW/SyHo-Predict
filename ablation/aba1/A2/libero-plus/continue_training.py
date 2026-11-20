"""Finish the current Stage I, then run Stage II and full LIBERO-Plus evaluation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import after_training_full_eval as evaluation

WORK = Path(__file__).resolve().parent
RUN = WORK / "protocol_plus_v1/seed_2026"
LOGS = WORK / "logs"
STATE = LOGS / "auto_continuation_status.json"
II_EXIT = LOGS / "stage_II_exit_code.txt"
CONFIG = json.loads((WORK / "config.json").read_text())
SETTINGS = CONFIG["a2_training"]
GPUS = SETTINGS["physical_gpus"]
PYTHON = CONFIG["model_python"]
read_json = evaluation.read_json
write_json = evaluation.write_json
_last_message = None
_last_print = 0


def exit_code(path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def commands():
    return {
        "stage_II": ["bash", str(WORK / "run_stage_ii_logged.sh")],
        "evaluation": [PYTHON, "-B", "-u", str(WORK / "after_training_full_eval.py"),
                       "--execute", "--workers", "8", "--poll-seconds", "10",
                       "--training-exit-file", str(II_EXIT)],
    }


def teacher_gate(pipeline, marker, code, live, files_ready):
    if pipeline.get("phase") == "failed" or code not in (None, 0):
        return "failed_stage_I"
    if live:
        return "waiting_stage_I_exit" if pipeline.get("phase") == "completed" else "waiting_stage_I"
    if code is None:
        return "failed_stage_I_missing_exit"
    if pipeline.get("phase") != "completed" or "I" not in pipeline.get("completed_stages", []):
        return "failed_stage_I_incomplete"
    if not files_ready or not marker.get("complete") or marker.get("stage") != "I" or marker.get("step") != SETTINGS["stage_I_steps"]:
        return "failed_stage_I_checkpoint"
    return "ready"


def teacher_snapshot():
    pipeline = read_json(RUN / "pipeline_status.json")
    launch = read_json(LOGS / "launch_record.json")
    states = [read_json(path) for path in (RUN / "I").glob("status_rank*.json")]
    live = [s["pid"] for s in states if evaluation.live_process(s.get("pid", 0), "a2_local.training")]
    launcher = launch.get("launcher_pid", 0)
    if evaluation.live_process(launcher, str(WORK / "run_logged.sh")):
        live.append(launcher)
    teacher = RUN / "I" / f"step_{SETTINGS['stage_I_steps']:06d}"
    code = exit_code(LOGS / "exit_code.txt")
    return dict(phase=teacher_gate(pipeline, read_json(teacher / "checkpoint_complete.json"), code,
                                   live, (teacher / "model.pt").is_file()),
                stage_I_exit_code=code, training_pids=live,
                stage_I_step=max([s.get("step", 0) for s in states] + [0]),
                pipeline_phase=pipeline.get("phase"), teacher=str(teacher))


def record(phase, **details):
    global _last_message, _last_print
    selected = read_json(RUN / "II" / "selected_runtime.json").get("world_size")
    from a2_stage2.runtime import POLICY
    row = dict(phase=phase, pid=os.getpid(), time=time.time(), armed=True, physical_gpus=POLICY["physical_gpus"] if "stage_II" in phase else GPUS,
               stage_II_physical_gpus=POLICY["physical_gpus"], evaluation_physical_gpus=[6],
               training_ranks=selected if "stage_II" in phase else SETTINGS["world_size"],
               stage_II_rank_policy=POLICY["rank_policy"], stage_II_ranks=selected, stage_II_max_ranks=POLICY["max_ranks"],
               stage_I_steps=SETTINGS["stage_I_steps"],
               stage_II_steps=SETTINGS["stage_II_steps"], evaluation_workers=8, evaluation_tasks=10030,
               **details)
    write_json(STATE, row)
    key = (phase, details.get("error"))
    if key != _last_message or time.monotonic() - _last_print >= 60:
        print(json.dumps(row, ensure_ascii=False), flush=True)
        _last_message, _last_print = key, time.monotonic()


def verify_teacher():
    sys.path[:0] = [str(WORK), str(WORK.parents[3])]
    from ablation.aba1.plus_common.config import load_config
    from a2_local.preflight import check_complete, check_teacher
    cfg = load_config(WORK / "config.json", "A2")
    if cfg != CONFIG:
        raise ValueError("Config changed after arming; inspect before continuing")
    teacher = RUN / "I" / f"step_{SETTINGS['stage_I_steps']:06d}"
    meta, marker = check_complete(teacher, verify_hash=True)
    check_teacher(teacher, cfg)
    write_json(LOGS / "stage_II_teacher_handoff.json", dict(time=time.time(), teacher=str(teacher),
               teacher_step=meta["step"], model_sha256=marker["files"]["model.pt"]["sha256"]))


def wait_for_teacher(poll):
    while True:
        current = teacher_snapshot()
        phase = current.pop("phase")
        record(phase, **current)
        if phase.startswith("failed"):
            raise RuntimeError(f"Stage I cannot continue: {phase}; inspect logs/train_gpu6.log")
        if phase == "ready":
            break
        time.sleep(poll)
    record("verifying_teacher", **current)
    verify_teacher()
    if teacher_snapshot()["phase"] != "ready":
        raise RuntimeError("Stage I state changed during teacher verification")
    (LOGS / "stage_I_exit_code.txt").write_text("0\n")


def run_child(command, phase, lock_fd, poll):
    record("starting_" + phase, command=command)
    # The child wrapper retains the lock if this monitor exits unexpectedly.
    # Its independent exit receipt prevents treating Stage I's exit as Stage II success.
    with open(os.devnull) as stdin:
        process = subprocess.Popen(command, cwd=WORK, stdin=stdin, start_new_session=True,
                                   pass_fds=(lock_fd,))
    while process.poll() is None:
        details = dict(child_pid=process.pid, command=command)
        if phase == "stage_II":
            current = evaluation.snapshot(II_EXIT)
            details.update(admission_phase=read_json(WORK / "logs/stage_II_adaptive_status.json").get("phase"),
                           stage_II_step=current["completed_steps"]["II"],
                           training_pids=current["training_worker_pids"])
        else:
            current = read_json(evaluation.STATE)
            details.update(evaluation_phase=current.get("phase"), completed=current.get("completed", 0),
                           eval_pid=current.get("eval_pid"))
        record(phase, **details)
        time.sleep(poll)
    if process.returncode != 0:
        raise RuntimeError(f"{phase} exited with code {process.returncode}; inspect its log")


def continue_run(lock_fd, poll):
    plan = commands()
    receipt = exit_code(II_EXIT)
    if II_EXIT.exists():
        if receipt != 0:
            raise RuntimeError("Stage II has a failed/invalid exit receipt; inspect before retrying")
    else:
        if (RUN / "II" / "arguments.json").exists() or read_json(RUN / "pipeline_status.json").get("stage") == "II":
            raise RuntimeError("An incomplete Stage II already exists; refusing a duplicate/restart")
        wait_for_teacher(poll)
        run_child(plan["stage_II"], "stage_II", lock_fd, poll)
        if exit_code(II_EXIT) != 0:
            raise RuntimeError("Stage II did not write a successful independent exit receipt")
    pipeline = read_json(RUN / "pipeline_status.json")
    if "II" not in pipeline.get("completed_stages", []) or evaluation.snapshot(II_EXIT)["phase"] != "ready":
        raise RuntimeError("Stage II final checkpoint/exit is not ready for evaluation")
    # This watcher verifies teacher/student hashes and budgets before launching inference.
    run_child(plan["evaluation"], "full_evaluation", lock_fd, poll)
    result = read_json(evaluation.STATE)
    if result.get("phase") != "completed" or result.get("completed") != 10030:
        raise RuntimeError("Full evaluation did not complete all 10030 tasks")
    record("completed", completed=10030, successes=result.get("successes"),
           success_rate=result.get("success_rate"), evaluation_output=result.get("evaluation_output"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args()
    if not 2 <= args.poll_seconds <= 60:
        parser.error("Polling must be 2..60 seconds")
    if not args.execute:
        print(json.dumps(dict(teacher_snapshot(), armed=False, commands=commands(),
                              stage_II_steps=SETTINGS["stage_II_steps"], evaluation_tasks=10030), indent=2))
        return
    with (LOGS / ".auto_continuation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("An automatic continuation or its child already owns the lock")
        try:
            launch = read_json(LOGS / "launch_record.json")
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            if launch.get("boot_id") != boot:
                raise RuntimeError("Server rebooted after this training launch; inspect before continuing")
            if SETTINGS["world_size"] != 8 or SETTINGS["processes_per_gpu"] * len(GPUS) != 8 or len(set(GPUS)) != len(GPUS):
                raise ValueError("This continuation requires eight ranks across the configured GPUs")
            from a2_stage2.runtime import POLICY
            write_json(LOGS / "auto_continuation_launch.json", dict(time=time.time(), pid=os.getpid(),
                       sid=os.getsid(0), boot_id=boot, physical_gpus=GPUS, training_ranks=8,
                       stage_I_steps=SETTINGS["stage_I_steps"], stage_II_steps=SETTINGS["stage_II_steps"],
                       evaluation_tasks=10030, evaluation_workers=8, launch_method="nohup",
                       stage_II_rank_policy=POLICY["rank_policy"], stage_II_max_ranks=POLICY["max_ranks"],
                       stage_II_physical_gpus=POLICY["physical_gpus"], stage_II_base_rank_to_gpu=POLICY["base_rank_to_gpu"],
                       config_sha256=hashlib.sha256((WORK / "config.json").read_bytes()).hexdigest(),
                       commands=commands(), training_restart=False))
            continue_run(lock.fileno(), args.poll_seconds)
        except Exception as error:
            record("failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
