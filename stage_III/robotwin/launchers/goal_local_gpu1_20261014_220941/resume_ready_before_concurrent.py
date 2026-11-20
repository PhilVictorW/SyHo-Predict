#!/usr/bin/env python3
"""离线检查与显式指定 GPU 的续跑入口；默认不启动进程。"""
import argparse
import datetime
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from stage_III.robotwin.common import (atomic_json, claim_run, load_config,
    read_json, sha256, validate_base_snapshot)
from stage_III.robotwin.make_manifests import validate_manifest, assert_disjoint
from stage_III.robotwin.workflow import commands
from stage_III.robotwin.runtime import child_env, spawn

RECORD = read_json(HERE.parent / "latest_gpu1.json")
RUN = Path(RECORD["run_dir"])
CONFIG = Path(RECORD["config"])
STUDENT = RUN / "interaction/checkpoints/student_000200.pt"
LABEL = dict(prepare_train="准备训练场景", prepare_val="准备验证场景",
    prepare_test="准备测试场景", paired_validation="配对在线验收",
    admission="Stage III 准入检查", stage_III="Stage III 奖励训练",
    final_comparison="最终独立测试")

def say(message):
    print(datetime.datetime.now().strftime("[%m-%d %H:%M:%S] ") + message, flush=True)

def completed(name):
    if name.startswith("prepare_"):
        p = RUN / ("manifests/prepared_" + name[8:] + ".json")
        if p.exists() and read_json(p).get("complete"):
            manifest = read_json(p)
            validate_manifest(manifest, name[8:])
            route = read_json(HERE / "gpu_route.json")
            return len(manifest["requested_cells"]) == route.get("task_mode_cells", 16)
    paths = {"paired_validation": ("paired_validation/report.json", "complete"),
             "admission": ("admission.json", "passed"),
             "final_comparison": ("final_comparison/summary.json", "complete")}
    if name in paths:
        rel, key = paths[name]
        return (RUN / rel).exists() and read_json(RUN / rel).get(key) is True
    if name == "stage_III":
        p = RUN / "rl/status.json"
        return p.exists() and read_json(p).get("phase") == "completed_pending_final_comparison"
    return False

def plan(gpu):
    cfg = load_config(CONFIG)
    scope = read_json(HERE / "gpu_route.json").get("task_mode_cells", 16)
    args = SimpleNamespace(run_dir=str(RUN), config=str(CONFIG), gpu=gpu,
                           cells=scope, final_evaluation=True)
    result = []
    for name, cmd in commands(args, cfg)[4:]:
        if completed(name): continue
        route_path = HERE / "gpu_route.json"
        route = read_json(route_path) if route_path.exists() else {}
        if name.startswith("prepare_") and route.get("scene_workers", 1) > 1:
            cmd = [cmd[0], "-u", str(HERE / "prepare_parallel.py"), *cmd[3:], "--workers", str(route["scene_workers"])]
        if "--gpu" in cmd: cmd[cmd.index("--gpu") + 1] = str(phase_gpu(name, gpu))
        result.append((name, cmd))
    return result

def phase_gpu(name, fallback):
    path = HERE / "gpu_route.json"
    if not path.exists(): return fallback
    route = read_json(path)
    gpu = route["training_gpu" if name in ("stage_III", "final_comparison") else "checks_gpu"]
    if type(gpu) is not int or not 0 <= gpu <= 7: raise ValueError("Invalid GPU route")
    return gpu

def process_identity(pid):
    try:
        path = Path("/proc") / str(pid)
        fields = (path / "stat").read_text().split(") ", 1)[1].split()
        return dict(pid=pid, start_ticks=fields[19], state=fields[0],
                    uid=path.stat().st_uid, pgid=int(fields[2]),
                    command=(path / "cmdline").read_bytes().replace(bytes([0]), b" ").decode())
    except (OSError, ValueError, IndexError): return None

class AdoptedPhase:
    """Existing owned phase; its completion must also pass artifact checks."""
    def __init__(self, record):
        self.record = record; self.pid = record["pid"]
        now = process_identity(self.pid)
        if now and now["state"] != "Z":
            if any(now[k] != record[k] for k in ("pid", "start_ticks", "uid", "pgid", "command")):
                raise ValueError("Adopted phase process identity changed")
            if now["uid"] != os.getuid() or now["pgid"] != self.pid:
                raise ValueError("Refusing to adopt a process outside this owned group")
    def poll(self):
        now = process_identity(self.pid)
        return None if now and now["start_ticks"] == self.record["start_ticks"] and now["state"] != "Z" else 0
    def wait(self, timeout=None):
        start = time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic() - start >= timeout:
                raise subprocess.TimeoutExpired(self.record["command"], timeout)
            time.sleep(.5)
        return 0  # Adopted exit code unavailable; completed(name) is mandatory.


def audit():
    from stage_III.robotwin.check import check
    cfg = load_config(CONFIG)
    say("正在进行离线配置、路径与源文件检查；不启动GPU。")
    static = check(CONFIG)
    validate_base_snapshot(cfg, read_json(RUN / "interaction/base_snapshot.json"))
    say("正在读取交互200步权重并核对SHA256（约2 GiB磁盘读取，无GPU计算）。")
    digest = sha256(STUDENT)
    assert read_json(str(STUDENT) + ".complete.json")["sha256"] == digest
    sensitivity = read_json(RUN / "sensitivity.json")
    assert sensitivity["passed"] and sensitivity["gate_nonzero"]
    assert sensitivity["student_sha256"] == digest
    pre = read_json(RUN / "interaction/preflight.json")
    assert pre["frozen_passed"] and pre["step0_max_abs"] <= cfg["acceptance"]["identity_atol"]
    assert pre["teacher_to_baseline_action_loss_ratio"] <= cfg["interaction"]["teacher_screen_ratio"]
    candidates = read_json(RUN / "manifests/candidates.json")["rows"]
    assert len({r["reset_seed"] for r in candidates}) == len(candidates)
    scenes = {}; physical = set(); complete = []
    scope = read_json(HERE / "gpu_route.json").get("task_mode_cells", 16)
    for split, quota in (("train", scope*2), ("val", scope*3), ("test", scope*3)):
        p = RUN / f"manifests/prepared_{split}.json"
        m = read_json(p) if p.exists() else {"rows": [], "events": [], "complete": False}
        for row in m["rows"]:
            assert row["prepared"] and row["split"] == split and row["instruction"]
            key = (row["task"], row["initial_state_hash"])
            assert key not in physical, "实际初态重复或跨集合泄漏"
            physical.add(key)
        if m["complete"]:
            validate_manifest(m, split); complete.append(m)
        scenes[split] = dict(accepted=len(m["rows"]), target=quota,
                            attempted=len(m["events"]), complete=m["complete"])
    if complete: assert_disjoint(*complete)
    report = dict(status="offline_ready_pending_gpu_checks", checked_at=time.time(),
        gpu_started=False, auto_resume=False, explicit_gpu_required=True,
        student=str(STUDENT), student_sha256=digest,
        source_files_unchanged=len(read_json(RUN / "interaction/base_snapshot.json")["code"]),
        static_status=static["status"], preflight_passed=True, sensitivity_passed=True,
        candidate_seeds_disjoint=True, prepared_physical_states_disjoint=True,
        scenes=scenes, pending=[LABEL[n] for n, _ in plan("指定卡号")],
        gpu_binding="命令行 --gpu 与 CUDA_VISIBLE_DEVICES 决定实际卡号；保留原配置以兼容检查点身份",
        note="剩余仿真准备与在线验收需要GPU；离线检查通过不等于Stage III准入通过。")
    atomic_json(HERE / "readiness.json", report)
    return report

def stop_tree(proc):
    """取消时只停止当前入口启动的子树，不触碰其他会话。"""
    if proc.poll() is not None: return
    os.killpg(proc.pid, signal.SIGSTOP)
    rows = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit(): continue
        try:
            fields = (p / "stat").read_text().split(") ", 1)[1].split()
            rows[int(p.name)] = (int(fields[1]), int(fields[2]))
        except (OSError, ValueError, IndexError): pass
    owned = {proc.pid}
    while True:
        new = owned | {pid for pid, (parent, _) in rows.items() if parent in owned}
        if new == owned: break
        owned = new
    groups = {rows[p][1] for p in owned if p in rows}
    assert all(g in owned for g in groups)
    for sig in (signal.SIGTERM, signal.SIGCONT):
        for g in groups:
            try: os.killpg(g, sig)
            except ProcessLookupError: pass
    try: proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        for g in groups:
            try: os.killpg(g, signal.SIGKILL)
            except ProcessLookupError: pass
        proc.wait(timeout=10)

def worker(gpu, adopt_path=None):
    lock = claim_run(RUN)
    cfg = load_config(CONFIG)
    adopted = read_json(adopt_path) if adopt_path else None
    if adopted and (adopted["run_dir"] != str(RUN) or adopted["phase"] != "prepare_train"):
        raise ValueError("Unexpected adoption record")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    atomic_json(HERE / "active_execution.json", dict(gpu=gpu, pid=os.getpid(),
        training_gpu=phase_gpu("stage_III", gpu), started_at=time.time(),
        authorization="用户指定检查后切换到GPU5训练" if adopted else "显式 --gpu N --execute"))
    for name, cmd in plan(gpu):
        selected_gpu = phase_gpu(name, gpu)
        if "--gpu" in cmd: cmd[cmd.index("--gpu") + 1] = str(selected_gpu)
        if name == "stage_III" and (RUN / "rl/last_checkpoint.json").exists():
            cmd += ["--resume", read_json(RUN / "rl/last_checkpoint.json")["path"]]
        say("开始/接续：" + LABEL[name] + f"；物理 GPU{selected_gpu}；奖励训练指定GPU{phase_gpu('stage_III', gpu)}")
        active = read_json(HERE / "active_execution.json")
        active.update(gpu=selected_gpu, phase=name, training_gpu=phase_gpu("stage_III", gpu))
        atomic_json(HERE / "active_execution.json", active)
        atomic_json(RUN / "workflow_status.json", dict(phase=name, status="running", gpu=selected_gpu,
            training_gpu=phase_gpu("stage_III", gpu), time=time.time()))
        if adopted and adopted["phase"] == name:
            if adopted["gpu"] != selected_gpu: raise ValueError("Cannot move an in-flight phase to another GPU")
            proc = AdoptedPhase(adopted["process"])
            say(f"接管正在运行的场景检查PID {proc.pid}，不重启当前任务。")
        else:
            proc = spawn(cmd, RUN / (name + ".log"), ROOT, child_env(cfg, selected_gpu, RUN))
        try:
            code = proc.wait()
            if adopted and adopted.get("expand_after_current_phase") and adopted["phase"] == name and not completed(name):
                say("当前检查批次结束，复用已验收场景，继续补齐全部100个任务模式。")
                proc = spawn(cmd, RUN / (name + ".log"), ROOT, child_env(cfg, selected_gpu, RUN))
                code = proc.wait()
        except BaseException:
            stop_tree(proc)
            atomic_json(RUN / "workflow_status.json", dict(phase=name, status="paused_by_user", time=time.time()))
            raise
        if code or not completed(name):
            atomic_json(RUN / "workflow_status.json", dict(phase=name, status="stopped", returncode=code,
                note="阶段异常、覆盖未完成或验收未通过；未启动后续阶段", time=time.time()))
            raise SystemExit("阶段未通过：" + LABEL[name] + "；请查看 " + str(RUN / (name + ".log")))
        say("完成：" + LABEL[name])
    atomic_json(RUN / "workflow_status.json", dict(phase="full_completed", status="comparison_completed", time=time.time()))
    say("全量流程结束。请同时核对实际轨迹覆盖、跳过记录和独立测试成绩。")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int); p.add_argument("--execute", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--adopt-state", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.execute and (a.gpu is None or not 0 <= a.gpu <= 7):
        p.error("执行必须明确指定 --gpu 0..7；默认仅离线检查。")
    if a.worker:
        if not a.execute: p.error("后台入口也必须显式 --execute")
        worker(a.gpu, a.adopt_state); return
    report = audit()
    say("离线核对通过：200步权重校验、41份源文件身份、冻结审计、梯度和门控检查。")
    for split, row in report["scenes"].items():
        say(f"{split} 场景准备：{row['accepted']}/{row['target']}")
    for name, cmd in plan(a.gpu if a.gpu is not None else "N"):
        say("待执行：" + LABEL[name]); print(shlex.join(cmd), flush=True)
    if not a.execute:
        say("仅完成离线检查；没有启动GPU、仿真或训练。等待用户指定GPU。")
        return
    session = f"robotwin_stageIII_resume_gpu{a.gpu}"
    probe = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True)
    if probe.returncode == 0: raise SystemExit("该后台会话已存在，请检查后再续跑，避免重复启动：" + session)
    cmd = [load_config(CONFIG)["runtime"]["model_python"], "-u", str(Path(__file__).resolve()),
           "--gpu", str(a.gpu), "--execute", "--worker"]
    shell = shlex.join(cmd) + " >> " + shlex.quote(str(HERE / f"resume_gpu{a.gpu}.log")) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", "workflow", "-c", str(ROOT), shell], check=True)
    say("后台已启动：" + session + "；先补齐场景和在线验收，再进入奖励训练。")

if __name__ == "__main__": main()
