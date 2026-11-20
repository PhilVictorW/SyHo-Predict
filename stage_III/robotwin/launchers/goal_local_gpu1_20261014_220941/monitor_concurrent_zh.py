"""只读 Stage III 紧凑看板，随终端尺寸调整。"""
import datetime, math, shutil, subprocess, sys, time, unicodedata
from pathlib import Path
from monitor_zh import read, jsonl, spark, alive

HERE = Path(__file__).resolve().parent

def cells(text):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)

def fit(text, width):
    text = " ".join(str(text).splitlines())
    if cells(text) <= width:
        return text
    result = ""
    for char in text:
        if cells(result + char) > max(0, width - 1):
            break
        result += char
    return result + ("…" if width else "")

def trend(values, width):
    values = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(x)]
    if not values:
        return "暂无记录"
    width = max(1, min(48, width))
    if len(values) > width:
        groups = [values[i*len(values)//width:(i+1)*len(values)//width] for i in range(width)]
        values = [sum(group)/len(group) for group in groups]
    return spark(values)

def render(width=None, height=None):
    terminal = shutil.get_terminal_size(fallback=(80, 20))
    width = max(1, (terminal.columns if width is None else width) - 1)
    height = max(1, (terminal.lines if height is None else height) - 1)
    record = read(HERE.parent / "latest_gpu1.json")
    rl = Path(record["run_dir"]) / "rl"
    status = read(rl / "status.json")
    metrics = jsonl(rl / "metrics.jsonl")
    last = metrics[-1] if metrics else {}
    skips = jsonl(rl / "skipped_groups.jsonl")
    cp = read(rl / "last_checkpoint.json")
    service = read(rl / "model_service.json")
    gpu = status.get("gpu", service.get("gpu", 3))
    phase = status.get("phase", "等待启动")
    labels = {"loading": "加载模型", "collecting": "在线采样", "iteration_committed": "本轮已保存",
              "completed_pending_final_comparison": "训练完成，等待独立评估"}
    committed = cp.get("iteration", last.get("iteration", 0))
    total = status.get("total", 100)
    fraction = min(1, committed / max(1, total))
    bar_width = max(4, min(32, width - 32))
    filled = int(bar_width * fraction)
    mode = str(status.get("mode", "-")).removeprefix("demo_")
    progress = {}
    if phase == "collecting" and status.get("group"):
        index = status.get("iteration", 0) * 4 + status["group"] - 1
        progress = read(rl / "groups" / f"group_{index:06d}" / "progress.json")
    age = max(0, int(time.time() - progress.get("last_progress_at", time.time())))
    trainer = "运行" if alive(status.get("pid")) else "未运行"
    model = "运行" if alive(service.get("pid")) else "未运行"
    lines = [
        f"RoboTwin Stage III | GPU{gpu} | {datetime.datetime.now():%m-%d %H:%M:%S}",
        f"阶段：{labels.get(phase, phase)} | 训练进程：{trainer} | 模型：{model}",
        "已保存 [" + "#" * filled + "-" * (bar_width - filled) + f"] {committed}/{total}轮 ({fraction:.1%})",
        f"有效轨迹 {status.get('episodes', last.get('total_episodes', 0))}/1600"
        f" | Actor更新 {status.get('update_count', last.get('update_count', 0))} | 跳过 {len(skips)}组",
        f"当前任务：{status.get('task', '-')} / {mode}",
    ]
    if progress:
        lines.append(f"第{status.get('iteration', 0)+1}轮 第{status['group']}/4组 第{progress.get('trajectory', 0)+1}/4条"
                     f" | 本条环境步 {progress.get('env_steps', 0)} | {age}秒前更新")
    else:
        lines.append(f"累计环境步 {status.get('env_steps', last.get('total_env_steps', 0))}")
    episodes = last.get("total_episodes", 0)
    successes = sum(row.get("successes", 0) for row in metrics)
    rate = f"{successes}/{episodes} ({successes/episodes:.1%})" if episodes else "暂无记录"
    lines.append(f"已提交采样成功率 {rate}（训练集）")
    for key, title in (("loss", "Actor loss"), ("reward_mean", "平均奖励"), ("kl", "KL"), ("eta_mean", "eta")):
        values = [row[key] for row in metrics if row.get(key) is not None]
        value = f"{values[-1]:.5g}" if values else "--"
        prefix = f"{title}" + " " * max(1, 12 - cells(title)) + f"{value:>11} "
        lines.append(prefix + trend(values, width - cells(prefix)))
    lines.append("最新权重：" + Path(cp.get("path", "尚未保存")).name)
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=2)
        used, free, util = [float(x.strip()) for x in result.strip().split(",")]
        lines.append(f"GPU{gpu} 显存 {used/1024:.1f}GiB | 剩余 {free/1024:.1f}GiB | 利用率 {util:.0f}%")
    except (OSError, ValueError, subprocess.SubprocessError):
        lines.append(f"GPU{gpu} 状态暂不可读")
    failure = read(rl / "failure.json")
    if failure:
        lines.insert(2, "训练异常：" + str(failure))
    lines += ["曲线覆盖已提交各轮；采样超时跳过，40分钟无进展跳过。",
              "每5秒刷新 | Ctrl+b 然后 d 退出查看，训练继续"]
    return "\n".join(fit(line, width) for line in lines[:height])

if __name__ == "__main__":
    if "--once" in sys.argv:
        print(render())
    else:
        print("\033[2J", end="", flush=True)
        while True:
            print("\033[2J\033[H" + render(), end="", flush=True)
            time.sleep(5)
