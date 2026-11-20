"""Read-only per-rank dashboard. No torch import, CUDA allocation or training writes."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

WORK = Path(__file__).resolve().parent
RUN = WORK / "protocol_plus_v1/seed_2026"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def read_metrics(stage):
    try:
        lines = (RUN / stage / "metrics.jsonl").read_text().splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def process_info(pid, stage):
    result = {"alive": False, "state": "-", "cpu": 0, "io": 0, "start": 0}
    if not pid:
        return result
    p = Path("/proc") / str(pid)
    try:
        args = (p / "cmdline").read_bytes().split(b"\0")
        fields = (p / "stat").read_text().rsplit(") ", 1)[1].split()
        expected = str(WORK / "config.json").encode()
        matches_stage = b"--stage" in args and args[args.index(b"--stage") + 1] == stage.encode()
        original = b"a2_local.training" in args and expected in args and b"--worker" in args and matches_stage
        adaptive = stage == "II" and (b"a2_stage2.worker" in args or b"a2_stage2_fixed8.worker" in args) and str(WORK).encode() in b" ".join(args)
        result.update(alive=(original or adaptive) and fields[0] not in ("Z", "X"),
                      state=fields[0], cpu=int(fields[11]) + int(fields[12]), start=int(fields[19]))
        try:
            io = dict(line.split(":", 1) for line in (p / "io").read_text().splitlines())
            result["io"] = int(io["read_bytes"]) + int(io["write_bytes"])
        except (OSError, ValueError, KeyError):
            pass
    except (OSError, ValueError, IndexError):
        pass
    return result


def age_text(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "--"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def progress_bar(step, total, width):
    ratio = min(1.0, max(0.0, step / max(1, total)))
    n = min(width, int(ratio * width))
    body = "#" * n
    if 0 < ratio < 1 and n < width:
        body += ">"
    return "[" + body.ljust(width, "-") + f"] {step:4d}/{total} {ratio:6.2%}"


class Tracker:
    def __init__(self):
        self.previous = {}
        self.initial_steps = {}

    def observe(self, stage, rank, state, proc, latest, now, active_seconds, stale_seconds, total):
        global_step = int(latest.get("step", 0))
        completed = max(int(state.get("step", 0)), global_step)
        key = (stage, rank, state.get("pid"), proc.get("start", 0))
        signature = (completed, state.get("micro"), state.get("epoch"), state.get("phase"))
        previous = self.previous.get(key)
        stamp = max(float(state.get("time", 0)), float(latest.get("time", 0)))
        changed = previous is not None and signature != previous["signature"]
        last_change = now if changed else previous["last_change"] if previous else min(now, stamp)
        activity = previous is not None and (proc.get("cpu", 0) > previous["cpu"]
                                             or proc.get("io", 0) > previous["io"])
        self.previous[key] = dict(signature=signature, last_change=last_change,
                                  cpu=proc.get("cpu", 0), io=proc.get("io", 0))
        age = max(0, now - last_change) if last_change else None
        phase = state.get("phase", "waiting")
        color = "33"
        if phase == "completed" and completed >= total:
            label, color = "已完成", "32"
        elif phase == "failed":
            label, color = "训练失败", "31"
        elif not state.get("pid"):
            label = "尚无进程状态"
        elif not proc["alive"]:
            label, color = "进程已退出", "31"
        elif phase == "ready":
            label, color = "已就绪，等待全部进程", "32"
        elif age is not None and age > stale_seconds:
            label = "长时间无进度"
        elif age is not None and age <= active_seconds:
            label, color = "最近有推进", "32"
        else:
            label = "等待下一更新"
        return dict(completed=completed, label=label, color=color, age=age,
                    activity=activity, changed=changed)


class Dashboard:
    def __init__(self, interval=2, stale_seconds=None, color=False):
        self.tracker = Tracker()
        self.interval = interval
        self.stale_seconds = stale_seconds
        self.color = color

    def paint(self, value, color):
        return f"\033[{color}m{value}\033[0m" if self.color else value

    def render(self, now=None):
        now = time.time() if now is None else now
        settings = read_json(WORK / "config.json").get("a2_training", {})
        pipeline = read_json(RUN / "pipeline_status.json")
        launch = read_json(WORK / "logs/launch_record.json")
        completed_stages = pipeline.get("completed_stages", [])
        stage = pipeline.get("stage") or (completed_stages[-1] if completed_stages else launch.get("stage", "I"))
        if stage not in ("I", "II"):
            stage = "I"
        selection = read_json(RUN / stage / "selected_runtime.json")
        adaptive = read_json(WORK / "logs/stage_II_adaptive_status.json") if stage == "II" else {}
        gpus = adaptive.get("physical_gpus", selection.get("physical_gpus", settings.get("physical_gpus", [6])))
        gpu = ",".join(map(str,gpus))
        total = int(settings.get(f"stage_{stage}_steps", 3000 if stage == "I" else 2500))
        contract = read_json(RUN / stage / "contract.json")
        world = int(selection.get("world_size", contract.get("world", settings.get("world_size", 8))))
        admission_states = []
        admission_active = stage == 'II' and pipeline.get('phase') == 'starting'
        if admission_active and pipeline.get('admission'):
            control = Path(pipeline['admission'])
            if control.is_relative_to(WORK):
                candidates = [read_json(p) for p in control.glob('worker*.json')]
                candidate_pid = adaptive.get('candidate_pid')
                if adaptive.get('phase') == 'probing_rank' and candidate_pid and not any(r.get('pid') == candidate_pid for r in candidates):
                    candidates.append(dict(pid=candidate_pid, rank=adaptive['candidate_rank'],
                        physical_gpu=adaptive['candidate_gpu'], phase='importing', time=adaptive.get('time', now)))
                admission_states = sorted([row for row in candidates if process_info(row.get('pid'), stage)['alive']], key=lambda row: row.get('rank',0))
            world = len(admission_states)
        rows = read_metrics(stage)
        train = [r for r in rows if r.get("split") == "train"]
        latest = train[-1] if train else {}
        resume_step = int(launch.get("resume_step", 0)) if stage == launch.get("stage", "I") else 0
        step = max(resume_step, int(latest.get("step", 0)))
        progress_reference = latest or dict(step=resume_step, time=launch.get("time", now))
        self.tracker.initial_steps.setdefault(stage, step)
        seconds = [float(r["update_seconds"]) for r in train[-20:]
                   if isinstance(r.get("update_seconds"), (int, float)) and math.isfinite(r["update_seconds"])]
        median = statistics.median(seconds) if seconds else 120.0
        stale = self.stale_seconds if self.stale_seconds is not None else max(180.0, 3 * median)
        active = max(20.0, min(90.0, median))
        accumulation = ([0]*world if admission_active else selection.get("accumulation_per_rank", contract.get("accumulation_per_rank", [len(range(k, 32, world)) for k in range(world)])))
        width = max(8, min(24, shutil.get_terminal_size((100, 32)).columns - 62))
        states = admission_states if admission_active else [read_json(RUN / stage / f"status_rank{rank}.json") for rank in range(world)]
        if not admission_active and selection.get("pids"):
            rendezvous = selection.get("rendezvous", "").removeprefix("file://")
            control = Path(rendezvous).parent
            if control.is_relative_to(WORK):
                states = [s if s.get("pid") == selection["pids"][rank] else read_json(control / f"worker{rank}.json")
                          for rank, s in enumerate(states)]
        procs = [process_info(s.get("pid"), stage) for s in states]
        count = sum(p["alive"] for p in procs)
        status = pipeline.get("phase", "waiting")
        if status == "training" and count == 0 and any(s.get("pid") for s in states):
            status = "worker 全部退出"
        expected_world = 8 if admission_active else world
        output = [f"A2 LIBERO-Plus | GPU{gpu} | Stage {stage} | 存活 {count}/{expected_world}",
                  time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(now)) + f" | 刷新 {self.interval:g}s | pipeline: {status}",
                  "总进度 " + progress_bar(step, total, width),
                  (f"固定8进程逐个加载：当前存活 {world}/8；GPU2/5/6/7各2个。" if admission_active else f"本次监测累计推进 +{max(0, step - self.tracker.initial_steps[stage])} 步；{world} 个 rank 同步训练同一模型。")]
        if stage == 'II' and adaptive.get('phase') in ('probing_rank', 'rank_admitted', 'waiting_memory', 'capacity_limit'):
            candidate = adaptive.get('candidate_rank')
            if candidate is not None:
                output.append(f"逐个加载：R{candidate} / GPU{adaptive.get('candidate_gpu', '--')} / PID {adaptive.get('candidate_pid', '--')} / {adaptive.get('candidate_phase', 'starting')}")
            elif adaptive.get('admitted_ranks') is not None:
                output.append(f"已通过实测 {adaptive['admitted_ranks']} 个进程；正在检查剩余显存，继续按固定映射逐个加载。")
        if latest:
            mean = statistics.mean(seconds) if seconds else 0
            output += [f"最近完整更新 {age_text(now - latest['time'])}前 | loss {latest['loss']:.5f} | grad {latest.get('grad_norm', 0):.5f}"]
            if admission_active:
                output.append("本次尚无新增训练步；以上为旧训练记录，等待8个进程全部就绪。")
            else:
                output.append(f"最近{len(seconds)}步均耗时 {mean:.1f}s/步 | 剩余时间参考 {age_text(max(0, total-step)*mean)}")
        else:
            output.append(f"从第{resume_step}步 checkpoint 恢复，正在校验/加载，尚无新增更新。" if resume_step else "尚无 optimizer 更新记录，正在初始化或初始验证。")
        output.append("")
        labels = {"waiting_checkpoint_space": "等待磁盘空间", "waiting_model_load": "排队加载", "loading_data": "数据/同步初始化", "training": "训练", "loading": "初始化/验证", "validation": "验证", "checkpoint": "保存", "completed": "完成", "failed": "失败"}
        for rank, (state, proc) in enumerate(zip(states, procs)):
            observed = self.tracker.observe(stage, rank, state, proc, progress_reference, now, active, stale, total)
            cap = accumulation[rank] if rank < len(accumulation) else 0
            phase = state.get("phase", "waiting")
            micro = f"{state.get('micro', 0)}/{cap}" if phase == "training" and int(state.get("step", 0)) >= step else "--"
            mapping = selection.get("rank_to_gpu", contract.get("rank_to_gpu", []))
            fallback = mapping[rank] if rank < len(mapping) else gpus[min(len(gpus)-1, rank // int(settings.get("processes_per_gpu", 1)))]
            physical = state.get("physical_gpu", fallback)
            output.append(f"R{rank} G{physical} PID {str(state.get('pid', '--')):<7} " + progress_bar(observed['completed'], total, width) + f" | micro {micro}")
            life = f"存活({proc['state']})" if proc["alive"] else "未运行"
            activity = "CPU/I/O活跃" if observed["activity"] else "CPU/I/O无变化"
            output.append("  " + self.paint(observed["label"], observed["color"]) + f" | 未变 {age_text(observed['age'])} | {labels.get(phase, phase)} | {life} | {activity}")
            if state.get("error"):
                output.append("  " + str(state["error"])[:110])
            if rank == 0 and phase == "waiting_checkpoint_space":
                output.append(f"  磁盘可用 {state.get('checkpoint_free_bytes', 0)/2**30:.1f} GiB / 保存需要 {state.get('checkpoint_required_bytes', 0)/2**30:.1f} GiB；保留内存中的更新及旧检查点，30秒后重查。")
        flow = read_json(WORK / "logs/auto_continuation_status.json")
        if flow:
            from after_training_full_eval import live_process
            alive = live_process(flow.get("pid", 0), "resume_stage_ii_fixed8.py") and (Path("/proc") / str(flow.get("pid", 0)) / "cwd").resolve() == WORK
            phase = flow.get("phase", "unknown")
            phases = {"waiting_stage_I": "等待I完成", "waiting_stage_I_exit": "等待I退出",
                      "verifying_teacher": "校验教师", "starting_stage_II": "启动II",
                      "stage_II": "训练II", "starting_full_evaluation": "启动全量推理",
                      "full_evaluation": f"推理 {flow.get('completed', 0)}/10030",
                      "completed": "全流程完成", "failed": "失败，请看auto-flow日志"}
            life = "在线" if alive else "已结束" if phase == "completed" else "离线"
            output.append(f"自动续接 I3000 → II2500 → 全量10030 | {life} | {phases.get(phase, phase)}")
        else:
            output.append("")
        output += [f"依据：步数/micro/阶段变化；无变化超过 {age_text(stale)}会提示。",
                   "CPU/I/O 仅辅助；长时间无进度不等同确认卡死。",
                   "Ctrl-C 退出进度面板；nohup 训练和自动续接继续。"]
        height = shutil.get_terminal_size((100, 32)).lines
        # Keep the heading and every rank visible when an attached client has a short pane.
        for prefix in ("", "CPU/I/O", "本次监测累计推进", "最近20步", "依据：", "Ctrl-b"):
            if len(output) <= height:
                break
            output = [line for line in output if not (line == "" if prefix == "" else line.startswith(prefix))]
        return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--stale-seconds", type=float)
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0 or (args.stale_seconds is not None and args.stale_seconds <= 0):
        parser.error("Intervals must be positive")
    interactive = sys.stdout.isatty() and not args.once
    dashboard = Dashboard(args.interval, args.stale_seconds, interactive and not args.no_color)
    try:
        if interactive:
            sys.stdout.write("\033[?25l")
        while True:
            try:
                frame = dashboard.render()
            except (OSError, ValueError, KeyError, IndexError) as error:
                frame = f"读取监控数据暂时失败，将重试：{error}"
            if interactive:
                sys.stdout.write("\033[H" + "\033[K\n".join(frame.splitlines())  + "\033[K\033[J")
            else:
                sys.stdout.write(frame + "\n")
            sys.stdout.flush()
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if interactive:
            sys.stdout.write("\033[?25h\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
