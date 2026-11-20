#!/usr/bin/env python3
"""Live tmux dashboard for the full LIBERO-plus Stage-III run.

This monitor is read-only: it observes status.json, group_summary.jsonl,
trajectory files, rollout processes and nvidia-smi.  It never changes training
checkpoints or starts/stops training workers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


SLOT_GPUS = [1, 1, 3, 3, 6, 6, 7, 7]
SLOT_PORTS = list(range(29700, 29708))
GPU_ORDER = [1, 3, 6, 7]
GPU_TO_SLOTS = {gpu: [i for i, value in enumerate(SLOT_GPUS) if value == gpu] for gpu in GPU_ORDER}
PHASE_NAMES = {
    "resuming": "断点恢复",
    "initializing": "初始化",
    "starting_servers": "启动服务",
    "preparing_cache": "准备 SSD 权重副本",
    "collecting": "在线采样",
    "updating": "GateActor 更新",
    "batch_complete": "批次完成",
    "completed": "已完成",
    "completed_with_skips": "遍历完成（含超时跳过）",
    "failed": "失败",
}


def read_json(path: Path, fallback: dict | None = None) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {} if fallback is None else fallback


def read_summary(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        for line in path.read_text().splitlines():
            try:
                value = json.loads(line)
                if isinstance(value, dict) and "group_id" in value:
                    rows.append(value)
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return rows


def command_output(cmd: list[str], timeout: float = 3.0) -> str:
    try:
        return subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, timeout=timeout,
                              check=False).stdout
    except Exception:
        return ""


def parse_rollouts() -> list[dict]:
    output = command_output(["ps", "-eo", "pid=,etimes=,stat=,args="], timeout=3)
    rows = []
    for line in output.splitlines():
        if "stage_III/libero_plus/rollout.py" not in line:
            continue
        group = re.search(r"--group\s+(\d+)", line)
        port = re.search(r"--url\s+http://127\.0\.0\.1:(\d+)", line)
        pid = re.match(r"\s*(\d+)\s+(\d+)\s+(\S+)", line)
        if not (group and port and pid):
            continue
        rows.append({
            "pid": int(pid.group(1)),
            "elapsed": int(pid.group(2)),
            "state": pid.group(3),
            "group": int(group.group(1)),
            "port": int(port.group(1)),
        })
    return rows


def parse_gpu_stats() -> dict[int, dict]:
    output = command_output([
        "nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], timeout=4)
    result = {}
    for line in output.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            idx, used, total, util = (int(float(x)) for x in parts)
            result[idx] = {"used": used, "total": total, "util": util}
        except ValueError:
            continue
    return result


def format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "--"
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes:02d}m{seconds:02d}s"


def bar(value: int, total: int, width: int = 44) -> str:
    ratio = 0.0 if total <= 0 else min(1.0, max(0.0, value / total))
    filled = int(round(width * ratio))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def slot_for_group(group_id: int, groups_per_update: int, nslots: int) -> int:
    if groups_per_update <= 0:
        return group_id % nslots
    batch_start = (group_id // groups_per_update) * groups_per_update
    return (group_id - batch_start) % nslots


def trajectory_files(run_dir: Path) -> dict[int, set[int]]:
    found: dict[int, set[int]] = {}
    pattern = re.compile(r"group_(\d+)_traj_(\d+)\.json$")
    try:
        for path in (run_dir / "trajectories").glob("group_*_traj_*.json"):
            match = pattern.match(path.name)
            if match:
                found.setdefault(int(match.group(1)), set()).add(int(match.group(2)))
    except OSError:
        pass
    return found


def started_at(run_dir: Path) -> float:
    match = re.search(r"grpo_(?:round2_)?full\d+_(\d{8}_\d{6})$", run_dir.name)
    if match:
        try:
            value = dt.datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
            return value.timestamp()
        except ValueError:
            pass
    try:
        return run_dir.stat().st_ctime
    except OSError:
        return time.time()


def trim_event(line: str, limit: int = 145) -> str:
    line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line).strip()
    if len(line) > limit:
        return line[:limit - 3] + "..."
    return line


def latest_events(log_path: Path, count: int = 4) -> list[str]:
    try:
        with log_path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 131072))
            lines = stream.read().decode(errors="replace").splitlines()
    except OSError:
        return []
    useful = [trim_event(line) for line in lines if "[Stage III]" in line]
    return useful[-count:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true", help="只显示一次，用于检查面板")
    parser.add_argument("--console", type=Path, default=None,
                        help="全量训练控制台日志路径")
    args = parser.parse_args()
    run_dir = args.run.resolve()
    status_path = run_dir / "status.json"
    summary_path = run_dir / "group_summary.jsonl"
    console_path = args.console.resolve() if args.console else run_dir.parents[1] / "full_train_8x2_console.log"
    previous_metrics: dict[str, float] = {}
    metric_deltas: dict[str, float] = {}
    previous_update = None
    started = started_at(run_dir)

    while True:
        status = read_json(status_path)
        config = status.get("config", {})
        slot_gpus = config.get("gpus", SLOT_GPUS)
        slot_ports = config.get("ports", SLOT_PORTS)
        gpu_order = list(dict.fromkeys(slot_gpus))
        gpu_to_slots = {gpu: [i for i, value in enumerate(slot_gpus) if value == gpu] for gpu in gpu_order}
        total_groups = int(config.get("selected_groups", 8380))
        traj_per_group = int(config.get("trajectories_per_group", 4))
        total_traj = int(config.get("planned_online_trajectories", total_groups * traj_per_group))
        groups_per_update = int(config.get("groups_per_update", 32))
        summary = read_summary(summary_path)
        summary_ids = {int(row["group_id"]) for row in summary}
        valid_ids = {int(row["group_id"]) for row in summary if row.get("status") != "skipped_timeout"}
        skipped_ids = {int(row["group_id"]) for row in summary if row.get("status") == "skipped_timeout"}
        for path in (run_dir / "skipped_groups").glob("group_*.json"):
            record = read_json(path)
            if record.get("status") == "skipped_timeout":
                skipped_ids.add(int(record["group_id"]))
        raw = trajectory_files(run_dir)
        raw_new = {gid: indexes for gid, indexes in raw.items() if gid not in summary_ids and gid not in skipped_ids}
        status_done_groups = int(status.get("groups_completed", 0))
        status_done_traj = int(status.get("trajectories_completed", 0))
        summary_done_groups = len(valid_ids)
        base_groups = max(status_done_groups, summary_done_groups)
        base_traj = max(status_done_traj, summary_done_groups * traj_per_group)
        raw_traj = sum(len(indexes) for indexes in raw_new.values())
        done_groups = min(total_groups, base_groups + sum(len(v) == traj_per_group for v in raw_new.values()))
        done_traj = min(total_traj, base_traj + raw_traj)
        processed_groups = min(total_groups, done_groups + len(skipped_ids))
        phase = str(status.get("phase", "unknown"))
        metrics = status.get("last_metrics") or {}
        rollouts = parse_rollouts()
        rollouts_by_port: dict[int, list[dict]] = {}
        for row in rollouts:
            rollouts_by_port.setdefault(row["port"], []).append(row)
        gpu_stats = parse_gpu_stats()

        elapsed = max(0.0, time.time() - started)
        resumed = float(status.get("resumed_at", started))
        rate_elapsed = max(0.0, time.time() - resumed)
        rate_done = done_traj - int(status.get("resume_base_trajectories", 0))
        rate = rate_done / rate_elapsed if rate_done > 0 and rate_elapsed > 60 else None
        stale = []
        for row in rollouts:
            heartbeat = read_json(run_dir / "progress" / f"group_{row['group']:04d}.json")
            age = time.time() - float(heartbeat.get("updated_at", time.time() - row["elapsed"]))
            if age > float(heartbeat.get("timeout_seconds", 180)):
                stale.append(row["group"])
        trainer_pid = status.get("trainer_pid")
        trainer_missing = trainer_pid is not None and not Path(f"/proc/{trainer_pid}").exists()
        if stale or trainer_missing:
            rate = None
        eta = (total_traj - len(skipped_ids) * traj_per_group - done_traj) / rate if rate and rate > 0 else None
        percent = 100.0 * done_traj / total_traj if total_traj else 0.0

        width = min(110, shutil.get_terminal_size((106, 32)).columns)
        update_id = status.get("optimizer_updates", status.get("batch", 0))
        planned_batches = (total_groups + groups_per_update - 1) // groups_per_update
        terminal_ids = valid_ids | skipped_ids | {
            gid for gid, indexes in raw_new.items() if len(indexes) == traj_per_group
        }
        trainer_state = "已退出" if trainer_missing else ("运行中" if trainer_pid else "待确认")
        print("\033[2J\033[H", end="")
        print("Stage III LIBERO-plus | 第二轮全量 GRPO | 四卡各两路 | 只读监控")
        print(f"时间 {dt.datetime.now().strftime('%m-%d %H:%M:%S')} | "
              f"{PHASE_NAMES.get(phase, phase)} | 主训练{trainer_state} | 每 {args.interval:g} 秒刷新")
        print(f"总遍历 {bar(processed_groups, total_groups, 24)} "
              f"{processed_groups}/{total_groups} 组 ({100 * processed_groups / total_groups:.2f}%，含跳过)")
        print(f"已训练 {bar(status_done_traj, total_traj, 24)} "
              f"{status_done_traj}/{total_traj} 条 | 更新 {update_id} 次，批次 {status.get('batch', 0)}/{planned_batches}")
        speed = f"{rate:.2f}" if rate else "--"
        print(f"有效采样 {done_traj} 条 ({percent:.2f}%，含待更新) | "
              f"跳过 {len(skipped_ids)} 组 | {speed} 条/s | 预计剩余 {format_seconds(eta)}")
        if stale:
            print(f"提示：组 {stale} 已超过无进展上限，等待跳过；暂停 ETA 估计。")
        if trainer_missing and phase not in ("completed", "completed_with_skips"):
            print("警告：训练调度器已退出；监控刷新不代表训练仍在运行。")
        print("-" * width)

        # Counts follow the current slot assignment, including groups completed
        # before a GPU migration. This is task progress, not physical GPU history.
        planned_by_gpu = {gpu: 0 for gpu in gpu_order}
        for gid in range(total_groups):
            gpu = slot_gpus[slot_for_group(gid, groups_per_update, len(slot_gpus))]
            planned_by_gpu[gpu] += 1
        for gpu in gpu_order:
            stat = gpu_stats.get(gpu, {})
            used, total, util = stat.get("used", 0), stat.get("total", 0), stat.get("util", 0)
            card_done = 0
            slot_text = []
            for slot in gpu_to_slots[gpu]:
                slot_groups = [gid for gid in valid_ids
                               if slot_for_group(gid, groups_per_update, len(slot_gpus)) == slot]
                slot_raw = {gid: indexes for gid, indexes in raw_new.items()
                            if slot_for_group(gid, groups_per_update, len(slot_gpus)) == slot}
                card_done += len(slot_groups) * traj_per_group + sum(len(v) for v in slot_raw.values())
                active = rollouts_by_port.get(slot_ports[slot], [])
                if active:
                    current = []
                    for item in active:
                        have = len(raw.get(item["group"], set()))
                        heartbeat = read_json(run_dir / "progress" / f"group_{item['group']:04d}.json")
                        age = max(0, int(time.time() - float(heartbeat.get("updated_at", time.time()))))
                        detail = f" {heartbeat.get('phase', '')} 心跳{age}s" if heartbeat else ""
                        current.append(f"g{item['group']} {have}/{traj_per_group}" + detail)
                    slot_text.append(f"S{slot}@{slot_ports[slot]} " + ",".join(current))
                else:
                    waiting = "等待本批其他采样" if phase == "collecting" else "等待调度"
                    slot_text.append(f"S{slot}@{slot_ports[slot]} {waiting}")
            card_skipped = sum(slot_gpus[slot_for_group(gid, groups_per_update, len(slot_gpus))] == gpu
                               for gid in skipped_ids)
            card_processed = sum(slot_gpus[slot_for_group(gid, groups_per_update, len(slot_gpus))] == gpu
                                 for gid in terminal_ids)
            card_total = planned_by_gpu[gpu]
            card_percent = 100 * card_processed / card_total if card_total else 0
            print(f"GPU{gpu} {bar(card_processed, card_total, 16)} "
                  f"{card_processed}/{card_total}组 {card_percent:5.1f}% | "
                  f"有效 {card_done}条 | 跳过 {card_skipped}组")
            memory = f"{used/1024:.1f}/{total/1024:.0f}G" if total else "--"
            for index, text in enumerate(slot_text):
                suffix = f" | 显存{memory} 负载{util}%" if index == 0 else ""
                print(f"  {text}{suffix}")
        print("-" * width)

        if metrics:
            labels = [
                ("reward_mean", "reward均值", ".4f"),
                ("loss", "loss", ".6f"),
                ("kl_old_new", "KL", ".6f"),
                ("clip_fraction", "clip比例", ".4f"),
                ("actor_eta_mean", "eta均值", ".4f"),
                ("grad_norm", "梯度范数", ".4f"),
            ]
            current_metrics = {key: float(metrics[key]) for key, _, _ in labels if key in metrics}
            if update_id != previous_update:
                metric_deltas = {key: value - previous_metrics[key]
                                 for key, value in current_metrics.items() if key in previous_metrics}
                previous_metrics = current_metrics
                previous_update = update_id
            print(f"最近更新#{update_id} | 有效 {metrics.get('groups', '--')}组 | "
                  f"成功 {metrics.get('successes', '--')}/{metrics.get('trajectories', '--')}条 | "
                  f"本批跳过 {metrics.get('groups_skipped', 0)}组 | Δ为相邻更新变化")
            values = []
            for key, label, fmt in labels:
                if key not in current_metrics:
                    continue
                delta = f"(Δ{metric_deltas[key]:+.2g})" if key in metric_deltas else ""
                values.append(f"{label}={format(current_metrics[key], fmt)}{delta}")
            for i in range(0, len(values), 3):
                print(" | ".join(values[i:i + 3]))
        else:
            print("等待首次 GateActor 更新指标。")
        print(f"配置: lr={config.get('learning_rate', '--')} | clip={config.get('clip_eps', '--')} | "
              f"效率奖励权重={config.get('efficiency_weight', '--')} | {traj_per_group}条/组 | {groups_per_update}组/批")
        print("卡进度按当前槽位分配累计；跳过组不计奖励/梯度。")
        events = latest_events(console_path, count=1)
        if phase == "preparing_cache":
            events = latest_events(run_dir.parents[1] / "cache_prepare_console.log", count=1) or events
        if events:
            print("最新: " + trim_event(events[-1], limit=50))
        print("保持后台运行：Ctrl-b 再按 d | tmux attach -t stage3_plus_round2_progress")
        sys.stdout.flush()
        if args.once:
            return
        try:
            time.sleep(max(1.0, args.interval))
        except KeyboardInterrupt:
            print("\n监控已退出，训练进程未处理。")
            return


if __name__ == "__main__":
    main()
