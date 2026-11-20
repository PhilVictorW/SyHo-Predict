#!/usr/bin/env python3
"""Read-only live tmux dashboard for the A2 three-GPU pipeline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import curses
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

WORK = Path(__file__).resolve().parent


def read(path):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return {}


def tail_rows(path, size=32768):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            length = stream.tell()
            stream.seek(max(0, length-size))
            content = stream.read().decode(errors='replace').splitlines()
        if length > size: content = content[1:]
    except OSError: return []
    rows = []
    for line in content:
        try: rows.append(json.loads(line))
        except ValueError: pass
    return rows


def bar(value, total, width=24):
    fraction = max(0., min(1., value / total)) if total else 0.
    filled = int(width * fraction)
    tip = '>' if 0 < fraction < 1 else ''
    return '[' + '#' * filled + tip + '-' * (width-filled-len(tip)) + ']'


def duration(seconds):
    if seconds is None: return '--'
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def number(value, spec='.5f'):
    try: return format(float(value), spec)
    except (TypeError, ValueError): return '--'


def gpu_stats():
    boards = {}
    processes = {}
    try:
        visible = ','.join(map(str, read(WORK/'run_plan.json')['physical_gpus']))
        raw = subprocess.check_output(['nvidia-smi', '-i', visible,
            '--query-gpu=index,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, timeout=5)
        for row in raw.splitlines():
            index, used, total, util = [x.strip() for x in row.split(',')]
            boards[int(index)] = dict(used=float(used)/1024, total=float(total)/1024, util=int(util))
        raw = subprocess.check_output(['nvidia-smi', '-i', visible,
            '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=5)
        for row in raw.splitlines():
            pid, memory = [x.strip() for x in row.split(',')]
            try: processes[int(pid)] = float(memory)/1024
            except ValueError: pass
        return dict(boards=boards, processes=processes, time=time.time())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return dict(boards=boards, processes=processes, time=time.time(), error=str(error))


def alive(pid):
    if not pid: return False
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        return raw[raw.rfind(')')+2:][:1] != 'Z'
    except OSError: return False


def snapshot():
    plan = read(WORK/'run_plan.json')
    root = WORK/'runs'/plan['run_name']
    pipeline = read(root/'pipeline.json')
    stage = pipeline.get('stage', 'I')
    phase = pipeline.get('phase', 'waiting')
    evaluation = WORK/'evaluation'/root.name/'full_40x50_step2000'
    states = {name: read(root/f'stage_{name}'/'status.json') for name in ('I', 'II')}
    active = stage if stage in ('I', 'II') else 'II'
    state = states[active]
    metrics = [x for x in tail_rows(root/f'stage_{active}'/'metrics.jsonl') if x.get('split') == 'train']
    times = [x['seconds'] for x in metrics[-20:] if x.get('seconds', 0) > 0]
    average = sum(times)/len(times) if times else None
    episodes = successes = tasks = 0
    if stage == 'evaluation' or phase == 'completed':
        for path in (evaluation/'tasks').glob('*/episodes.checkpoint.jsonl'):
            records = {}
            for row in tail_rows(path, 1024*1024):
                if not row.get('error') and 0 <= row.get('episode_idx', -1) < 50:
                    records[row['episode_idx']] = row
            episodes += len(records)
            successes += sum(bool(row.get('success')) for row in records.values())
            tasks += len(records) == 50
    workers = []
    for rank, gpu in enumerate(plan['physical_gpus']):
        worker = read(root/f'stage_{active}'/f'worker_{rank}.json')
        heartbeat = read(root/f'stage_{active}'/f'heartbeat_{rank}.json')
        workers.append(dict(rank=rank, gpu=gpu, worker=worker, heartbeat=heartbeat,
                            micro_total=plan['rank_anchors_per_update'][rank]))
    return dict(plan=plan, root=root, pipeline=pipeline, phase=phase, stage=stage, active=active,
        state=state, states=states, workers=workers, metric=metrics[-1] if metrics else {},
        average=average, episodes=episodes, successes=successes, tasks=tasks,
        server=read(evaluation/'server_ready.json'))


def lines(data, stats, width=104):
    plan, stage, state, phase = data['plan'], data['stage'], data['state'], data['phase']
    now = time.time()
    content = []
    def add(text='', color=0): content.append((text, color))
    add(f"A2  GOAL-ONLY 16 QUERY | Stage {stage} | {phase.upper()} | {datetime.now():%m-%d %H:%M:%S}", 1)
    add('Synchronous training: completed updates are shared; micro-batches show each GPU separately.')
    a = data['states']['I'].get('step', 0)
    b = data['states']['II'].get('step', 0)
    add(f"Teacher {bar(a,1000,14)} {a:4}/1000   Student {bar(b,2000,14)} {b:4}/2000   Eval {data['episodes']:4}/2000", 2)
    metric = data['metric']
    if stage in ('I', 'II'):
        target = plan[f'stage_{stage}']['steps']
        completed = state.get('step', 0)
        remaining = (target-completed)*data['average'] if data['average'] else None
        add(f"Loss {number(metric.get('loss'))} | Flow {number(metric.get('flow'))} | GoalKD {number(metric.get('goal_distill'))} | Grad {number(metric.get('grad_norm'), '.3f')}")
        add(f"Avg {number(data['average'], '.1f')} s/update | elapsed {duration(state.get('elapsed_seconds'))} | stage ETA ~{duration(remaining)} | batch 32 = 11+11+10")
    else:
        add(f"Full LIBERO: {data['tasks']}/40 tasks | {data['episodes']}/2000 episodes | successes {data['successes']}")
        add('Evaluation uses Student step 2000 on GPU 1. Stage I / II weights are already saved.')
    add()
    for item in data['workers']:
        gpu, rank = item['gpu'], item['rank']
        heartbeat, worker = item['heartbeat'], item['worker']
        pid = worker.get('pid')
        board = stats.get('boards', {}).get(gpu, {})
        if stage in ('I', 'II'):
            completed = state.get('step', 0)
            target = plan[f'stage_{stage}']['steps']
            is_alive = alive(pid)
            current_step = heartbeat.get('step', 0)
            micro = heartbeat.get('micro', 0) if current_step > completed else 0
            micro = min(micro, item['micro_total'])
            age = max(0, now-heartbeat['time']) if heartbeat.get('time') else None
            activity = state.get('phase', 'initializing')
            if activity == 'training':
                activity = 'sync / optimizer' if micro == item['micro_total'] else ('backward' if micro else 'loading next batch')
            if phase == 'failed': activity = 'FAILED'
            elif pid and not is_alive and state.get('phase') != 'completed': activity = 'PROCESS EXITED'
            health = 'LIVE' if is_alive else ('DONE' if state.get('phase') == 'completed' else 'WAIT')
            add(f"GPU {gpu} / rank {rank}  {bar(completed,target,24)} {completed:4}/{target} {100*completed/target:5.1f}%  PID {pid or '--'} {health}", 3 if is_alive else 4)
            add(f"  update {max(completed+1, current_step) if completed < target else target} | micro {bar(micro,item['micro_total'],10)} {micro:2}/{item['micro_total']}  {activity:<20}  heartbeat {duration(age)}")
        else:
            pid = data['server'].get('pid') if gpu == plan['evaluation']['physical_gpu'] else None
            if pid:
                add(f"GPU {gpu} / inference  {bar(data['episodes'],2000,24)} {data['episodes']:4}/2000 episodes  PID {pid}", 3)
            else: add(f"GPU {gpu} / rank {rank}  {bar(2000,2000,24)} training complete", 2)
            add(f"  {'Inference running' if pid and alive(pid) else 'Training workers finished'}")
        job_mem = stats.get('processes', {}).get(pid)
        add(f"  A2 process VRAM {number(job_mem,'.2f')} GiB | entire GPU {number(board.get('used'),'.1f')}/{number(board.get('total'),'.0f')} GiB | entire GPU util {board.get('util','--')}%")
        add()
    if phase == 'failed':
        error = data['pipeline'].get('error', '').strip().splitlines()
        add(error[-1] if error else 'Pipeline failed: inspect train-logs window.', 5)
    cache_age = now-stats['time'] if stats.get('time') else None
    add(f"Refresh 1s | GPU stats every 10s (age {duration(cache_age)}) | q: close dashboard | tmux Ctrl-b 0: logs", 4)
    return content


def display(screen, interval):
    try: curses.curs_set(0)
    except curses.error: pass
    screen.nodelay(True)
    if curses.has_colors():
        curses.start_color()
        try: curses.use_default_colors()
        except curses.error: pass
        for index, color in enumerate([curses.COLOR_CYAN, curses.COLOR_BLUE, curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED], 1):
            curses.init_pair(index, color, -1)
    stats = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(gpu_stats)
        last_query = time.monotonic()
        while True:
            if future is not None and future.done():
                result = future.result()
                if not result.get('error') or not stats: stats = result
                future = None
            if future is None and time.monotonic()-last_query >= 10:
                future = pool.submit(gpu_stats)
                last_query = time.monotonic()
            height, width = screen.getmaxyx()
            screen.erase()
            try: rendered = lines(snapshot(), stats, width)
            except (OSError, KeyError, ValueError) as error: rendered = [(f'Waiting for state files: {error}', 4)]
            # At 20 terminal rows keep all three cards and omit blank separators.
            if len(rendered) > height: rendered = [x for x in rendered if x[0]]
            for y, (line, color) in enumerate(rendered[:height]):
                try:
                    style = curses.color_pair(color) if curses.has_colors() else 0
                    screen.addnstr(y, 0, line, max(1, width-1), style)
                except curses.error: pass
            screen.refresh()
            end = time.monotonic()+interval
            while time.monotonic() < end:
                if screen.getch() in (ord('q'), ord('Q')): return
                time.sleep(.1)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=1.)
    args = parser.parse_args()
    if args.once:
        print('\n'.join(line for line, _ in lines(snapshot(), gpu_stats())))
    else:
        curses.wrapper(display, max(.2, args.interval))


if __name__ == '__main__': main()
