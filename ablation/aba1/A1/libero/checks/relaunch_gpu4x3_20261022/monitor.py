#!/usr/bin/env python3
"""Read-only live A1 progress display; independent of the training workers."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import shutil
import unicodedata
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

WORK = Path(__file__).resolve().parent
PHASE = {'training':'训练中', 'validating':'验证中', 'saving':'保存权重', 'completed':'已完成',
         'failed':'失败', 'waiting_for_resources':'等待资源', 'pending':'尚未开始'}


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def rows(path):
    try:
        with path.open('rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size-131072))
            if size > 131072:
                f.readline()
            lines = f.read().decode(errors='replace').splitlines()
        result = []
        for line in lines:
            try:
                row = json.loads(line)
                if row.get('split') == 'train': result.append(row)
            except ValueError:
                pass
        return result
    except OSError:
        return []


def duration(seconds):
    if seconds is None: return '待估算'
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def bar(step, total, width=36):
    filled = min(width, int(width*step/max(total,1)))
    return '[' + '#' * filled + '-' * (width-filled) + f'] {step:4d}/{total} ({step/total:6.2%})'


def render(root):
    now = time.time()
    plan = read(WORK/'run_plan.json')
    pipeline = read(root/'pipeline.json')
    active = pipeline.get('stage','II' if pipeline.get('phase') == 'completed' else 'I')
    world = pipeline.get('world_size',plan.get('world_size',4))
    lines = ['A1 LIBERO  |  Local-only 16 tokens',
             datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y-%m-%d %H:%M:%S UTC+8') + '  |  每 5 秒刷新',
             f"GPU 5  |  当前 {world} 个训练进程  |  有效 batch 32  |  {PHASE.get(pipeline.get('phase'),'初始化/加载中')}", '']
    total_done = 0
    for stage, label, target in [('I','教师 Stage I',1000),('II','学生 Stage II',2000)]:
        folder = root/('stage_'+stage)
        state = read(folder/'status.json')
        metrics = rows(folder/'metrics.jsonl')
        last = metrics[-1] if metrics else {}
        # Count completed optimizer updates, never in-flight microbatches.
        step = max(int(state.get('step',0)),int(last.get('step',0)))
        total_done += step
        phase = state.get('phase','初始化/加载中' if stage == active and pipeline.get('phase')=='training' else 'pending')
        lines.append(f'{label}  {PHASE.get(phase,phase)}')
        lines.append(bar(step,target))
        if metrics:
            recent = [float(x['seconds']) for x in metrics[-20:] if x.get('seconds',0)>0]
            speed = statistics.mean(recent) if recent else None
            remaining = (target-step)*speed if speed is not None else None
            elapsed = state.get('elapsed_seconds',0)
            if phase in ('training','validating','saving'):
                elapsed += max(0,now-state.get('updated_at',now))
            lines.append(f"已用 {duration(elapsed)}  |  近 {len(recent)} 步 {speed:.1f} 秒/步  |  剩余约 {duration(remaining)}" if speed else f'已用 {duration(elapsed)}')
            losses = [f"{key}={last[key]:.6f}" for key in ['loss','flow','local_distill'] if key in last]
            lines.append('  '.join(losses))
            lines.append(f"lr={last.get('learning_rate',0):.3g}  |  grad_norm={last.get('grad_norm',0):.4f}  |  最近更新 {int(max(0,now-last.get('time',now)))} 秒前")
        saved = []
        for marker in sorted((folder/'checkpoints').glob('step_*/checkpoint_complete.json')):
            item = read(marker)
            if 'step' in item: saved.append(str(item['step']))
        lines.append('已保存权重：' + (', '.join(saved) if saved else '暂无'))
        lines.append('')
    lines += ['总步数（按 optimizer step 计，不代表耗时比例）',bar(total_done,3000),'']
    folder = root/('stage_'+active)
    lines.append('当前 rank 状态：')
    found = False
    for rank in range(world):
        worker = read(folder/f'worker_{rank}.json')
        beat = read(folder/f'heartbeat_{rank}.json')
        if not worker: continue
        found = True
        pid = worker.get('pid',0)
        alive = Path(f'/proc/{pid}').exists()
        share = len(range(rank,32,world))
        micro = f"step {beat.get('step','-')}，micro {beat.get('micro',0)}/{share}" if beat else '权重已加载'
        age = f"，心跳 {int(max(0,now-beat['time']))} 秒前" if 'time' in beat else ''
        lines.append(f"  rank {rank}  PID {pid}  {'运行中' if alive else '已退出'}  {micro}{age}")
    if not found: lines.append('  正在创建进程/加载权重；以训练日志为准')
    try:
        gpu = subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=3).strip().split(',')
        lines += ['',f'GPU 5 全卡：显存 {float(gpu[0])/1024:.1f}/{float(gpu[1])/1024:.0f} GiB，利用率 {gpu[2].strip()}%（包含其他任务）']
    except (OSError,ValueError,subprocess.SubprocessError):
        lines += ['', 'GPU 状态暂不可读；训练进度仍正常刷新']
    if pipeline.get('phase') == 'failed':
        lines += ['', '失败信息：'+pipeline.get('error','请查看 train-log 窗口').strip().splitlines()[-1]]
    lines += ['', '教师仅保存 1000；学生每 500 步保存，保留最近两份。',
              'ETA 按近期训练速度估算，不含验证、保存和 OOM 重试开销。',
              'Ctrl+b 然后 d：退出查看，训练与进度窗口继续后台运行。',
              'Ctrl+b 然后 0：训练日志；Ctrl+b 然后 1：本进度窗口。',
              '运行目录：'+str(root)]
    # Keep both stage bars visible even in a small attached terminal.
    lines = [line for line in lines if line and not line.startswith('运行目录：')]
    if sys.stdout.isatty():
        size = shutil.get_terminal_size((120, 40))
        for prefix in ('lr=', '教师仅保存', 'ETA 按', '当前 rank 状态：'):
            if len(lines) < size.lines: break
            lines = [line for line in lines if not line.startswith(prefix)]
        clipped=[]
        for line in lines:
            width=0; chars=[]
            for char in line:
                width += 2 if unicodedata.east_asian_width(char) in ('W','F') else 1
                if width >= size.columns-1: break
                chars.append(char)
            clipped.append(''.join(chars))
        lines=clipped
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--interval',type=float,default=5)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    root=args.run or WORK/'runs'/read(WORK/'run_plan.json')['run_name']
    if args.interval < 1: parser.error('--interval must be at least 1 second')
    try:
        while True:
            try:
                text=render(root)
            except Exception as error:
                text=f'进度读取暂时失败：{error}\n稍后自动重试；训练进程不受影响。'
            if sys.stdout.isatty() and not args.once:
                sys.stdout.write('\033[H\033[2J')
            print(text,flush=True)
            if args.once: break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
