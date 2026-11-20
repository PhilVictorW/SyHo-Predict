"""Read-only tmux dashboard for the current A0 LIBERO-Plus training/evaluation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import statistics
import subprocess
import time

WORK = Path(__file__).resolve().parent
RUN = WORK / 'runs/single_latent16_dino_step1000_2gpu07_20261019'
WORKFLOW = WORK / 'workflow_step1000_gpu02.json'
LOG = WORK / 'logs/resume_step750_gpu02_staggered_20261019.log'


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def bar(done, total):
    ratio = min(1., max(0., done / max(total, 1)))
    n = int(ratio * 32)
    return '[' + '#' * n + '-' * (32 - n) + f'] {done}/{total} ({ratio:.1%})'


def duration(seconds):
    if seconds is None:
        return '--'
    minutes, sec = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return (f'{hours}h ' if hours else '') + f'{minutes:02d}m {sec:02d}s'


def recent_metrics(since):
    try:
        with (RUN / 'metrics.jsonl').open('rb') as f:
            f.seek(max(0, f.seek(0, 2) - 512 * 1024))
            lines = f.read().decode(errors='replace').splitlines()
    except OSError:
        return []
    rows = {}
    for line in lines:
        try:
            row = json.loads(line)
            if row.get('split') == 'train' and row.get('time', 0) >= since:
                rows[row['step']] = row
        except (ValueError, KeyError):
            pass
    return sorted(rows.values(), key=lambda r: r['step'])


def render():
    workflow = read_json(WORKFLOW)
    status = read_json(RUN / 'status.json')
    resumed = read_json(RUN / 'resume_verified.json')
    rows = recent_metrics(resumed.get('time', workflow.get('time', float('inf'))))
    latest = rows[-1] if rows else {}
    step = int(status.get('step', resumed.get('step', 750)))
    speed = statistics.mean(r['update_seconds'] for r in rows[-30:]) if rows else None
    phase = status.get('phase', 'loading')
    final = read_json(RUN / 'final.json')
    marker = read_json(Path(final['path']) / 'checkpoint_complete.json') if final.get('path') else {}
    saved = bool(final.get('step') == 1000 and marker.get('step') == 1000 and marker.get('complete'))
    labels = {'training':'训练中', 'loading':'加载 / 恢复中', 'validation':'验证中',
              'checkpointing':'保存检查点中', 'completed':'训练完成', 'failed':'训练失败'}
    label = labels.get(phase, phase)
    if workflow.get('phase') == 'failed':
        label = '工作流程失败，请查看日志'
    elif workflow.get('phase') == 'evaluating_step1000':
        label = '1000 步已保存，推理中'
    elif workflow.get('phase') == 'completed':
        label = '训练及推理完成'
    elif not alive(workflow.get('pid')) and phase != 'completed':
        label += '（工作流程进程已退出）'
    result = [f'A0 LIBERO-Plus | GPU 0,2 | {time.strftime("%Y-%m-%d %H:%M:%S")}',
              '=' * 68, '状态：' + label, '累计训练  ' + bar(step, 1000),
              '本次续训  ' + bar(max(0, step - 750), 250)]
    if latest:
        result.append(f'Action loss: {latest["loss"]:.6f} | 最近30步均速: {speed:.2f} 秒/步')
    eta = (1000 - step) * speed if speed is not None and phase == 'training' else None
    result.append('剩余训练：' + duration(eta) + '（不含验证、保存及推理）')
    result.append('第1000步权重：' + ('已完整保存' if saved else '尚未完成保存'))
    checkpoint = read_json(RUN / 'latest.json')
    result.append(f'最近完整检查点：step {checkpoint.get("step", "--")}')
    for rank in range(2):
        startup = read_json(RUN / f'startup_rank{rank}.json')
        pid = startup.get('pid')
        load_label = {'waiting_for_load_turn':'等待前一卡恢复', 'loading_model':'加载模型',
                      'restoring_checkpoint':'恢复检查点', 'checkpoint_restored':'已恢复，等待另一卡',
                      'both_ranks_ready':'已就绪'}.get(startup.get('phase'), '--')
        result.append(f'训练 GPU {[0,2][rank]}：{load_label} | PID {pid or "--"} | '
                      + ('进程运行' if alive(pid) else '进程已退出'))
    try:
        query = subprocess.run(['nvidia-smi','--id=0,2',
            '--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3, check=True)
        for line in query.stdout.splitlines():
            gpu, used, total, util = [x.strip() for x in line.split(',')]
            result.append(f'GPU {gpu} 全卡显存：{int(used)/1024:.1f}/{int(total)/1024:.1f} GiB | 利用率 {util}%')
    except (OSError, ValueError, subprocess.SubprocessError):
        result.append('GPU 指标暂不可用')
    result.append(f'硬盘可用：{shutil.disk_usage(WORK).free / 2**30:.1f} GiB | 保存检查门槛：45 GiB')
    folder = RUN / 'evaluation/full_epoch_0000_step_000001000'
    dbpath = folder / 'queue.sqlite'
    if dbpath.exists():
        try:
            with sqlite3.connect(dbpath.as_uri() + '?mode=ro', uri=True, timeout=1) as db:
                states = dict(db.execute('SELECT state,COUNT(*) FROM tasks GROUP BY state'))
            done, total = states.get('done', 0), sum(states.values())
            result += ['推理进度  ' + bar(done, total), f'运行中 {states.get("running",0)} | 错误 {states.get("failed",0)}']
            evaluation = read_json(folder / 'results.json')
            if evaluation.get('complete') and evaluation.get('success_rate') is not None:
                result.append(f'最终成功率：{evaluation["success_rate"]:.2%}')
        except (sqlite3.Error, OSError):
            result.append('推理队列读取中')
    else:
        result.append('推理：等待第1000步权重保存完成')
    if workflow.get('error') or status.get('error'):
        result.append('错误：' + str(workflow.get('error') or status.get('error')))
    age = max(0, time.time() - status.get('time', time.time()))
    result += [f'训练状态距今 {int(age)} 秒 | 每5秒刷新', '=' * 68,
               'Ctrl+b 然后 d：离开面板，训练继续运行', '日志：logs/' + LOG.name]
    return '\n'.join(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    while True:
        try:
            screen = render()
        except Exception as error:
            screen = '面板读取暂时失败：' + repr(error)
        print(('' if args.once else '\033[H\033[2J') + screen, flush=True)
        if args.once:
            break
        time.sleep(5)
