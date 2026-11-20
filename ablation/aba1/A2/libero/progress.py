#!/usr/bin/env python3
"""Read-only live tmux dashboard for A2 training and parallel evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import curses
from datetime import datetime
import json
import os
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
        visible = ','.join(map(str, dict.fromkeys(read(WORK/'run_plan.json')['physical_gpus'])))
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


def process_info(pid, root):
    """Check identity as well as liveness, excluding recycled PIDs."""
    if not pid: return dict(alive=False, state='--', cpu_seconds=0.)
    try:
        folder = Path(f'/proc/{pid}')
        fields = (folder/'stat').read_text().rsplit(')', 1)[1].split()
        command = (folder/'cmdline').read_bytes().replace(bytes([0]), b' ').decode(errors='replace')
        matched = 'a2_goal.training' in command and str(root) in command
        return dict(alive=matched and fields[0] not in ('Z', 'X'), state=fields[0],
                    cpu_seconds=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),
                    started=fields[19])
    except (OSError, ValueError, IndexError):
        return dict(alive=False, state='--', cpu_seconds=0.)


def movement(item, data, now):
    """Liveness alone does not prove advancing optimizer/micro-batch work."""
    worker, heartbeat, proc = item['worker'], item['heartbeat'], item['process']
    state, metric = data['state'], data['metric']
    phase = state.get('phase', 'initializing')
    last_progress = max(heartbeat.get('time', 0), metric.get('time', 0))
    age = max(0., now-last_progress) if last_progress else None
    threshold = max(300., 3*(data['average'] or 0))
    if data['phase'] == 'failed' or item['failure'] or phase == 'failed':
        label, color = '训练失败', 5
    elif phase == 'completed':
        label, color = '已完成', 2
    elif worker.get('pid') and not proc['alive']:
        label, color = '进程退出', 5
    elif not worker.get('pid'):
        label, color = '初始化 / 等待进程记录', 4
    elif phase == 'initializing':
        label = {'waiting_for_load': '排队等待加载', 'loading_model': '正在加载模型', 'model_loaded': '模型就绪，等待同步'}.get(worker.get('phase'), '初始化中')
        color = 4
    elif phase in ('validating', 'saving'):
        label, color = ('验证中' if phase == 'validating' else '保存权重中'), 4
        if now-state.get('updated_at', now) > max(1200., threshold):
            label += '（长时间无新记录）'
    elif age is not None and age > threshold:
        label, color = '疑似停滞', 5
    elif heartbeat.get('step', 0) > state.get('step', 0) and heartbeat.get('micro', 0) >= item['micro_total']:
        label, color = '同步 / 更新中', 4
    elif age is not None and age <= 30:
        label, color = '持续推进', 3
    elif age is not None:
        label, color = '等待数据 / 计算', 4
    else:
        label, color = '初始化中', 4
    return dict(label=label, color=color, age=age, threshold=threshold)


class ProcessSampler:
    def __init__(self):
        self.previous = {}

    def update(self, data):
        now = time.monotonic()
        for item in data['workers']:
            proc = item['process']
            key = (str(data['root']), data['active'], item['worker'].get('pid'), proc.get('started'))
            previous = self.previous.get(key)
            proc['cpu_percent'] = (max(0., proc['cpu_seconds']-previous[1])*100/(now-previous[0])
                                   if previous and now > previous[0] else None)
            self.previous[key] = (now, proc['cpu_seconds'])
        for lane in data.get('evaluation_status', {}).get('workers', []):
            key = ('evaluation', str(data['root']), lane.get('slot'), lane.get('server_pid'))
            value = (lane.get('calls'), lane.get('suite'), lane.get('task_id'), lane.get('episodes_completed'), lane.get('phase'))
            previous = self.previous.get(key)
            changed = now if previous is None or previous[0] != value else previous[1]
            self.previous[key] = (value, changed)
            lane['movement_age'] = now-changed


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
        worker = read(root/f'stage_{active}'/f'worker_{rank}.json') or read(root/f'stage_{active}'/f'startup_rank_{rank}.json')
        heartbeat = read(root/f'stage_{active}'/f'heartbeat_{rank}.json')
        workers.append(dict(rank=rank, gpu=gpu, worker=worker, heartbeat=heartbeat,
                            process=process_info(worker.get('pid'), root),
                            failure=read(root/f'stage_{active}'/f'failure_rank_{rank}.json'),
                            micro_total=plan['rank_anchors_per_update'][rank]))
    return dict(plan=plan, root=root, pipeline=pipeline, phase=phase, stage=stage, active=active,
        state=state, states=states, workers=workers, metric=metrics[-1] if metrics else {},
        average=average, recent_steps=sum(x.get('time', 0) >= time.time()-300 for x in metrics),
        episodes=episodes, successes=successes, tasks=tasks,
        server=read(evaluation/'server_ready.json'), evaluation_status=read(evaluation/'status.json'),
        evaluation_settings=read(WORK/'evaluation_parallel.json'))


def lines(data, stats, width=104):
    plan, stage, state, phase = data['plan'], data['stage'], data['state'], data['phase']
    now = time.time()
    content = []
    def add(text='', color=0): content.append((text, color))
    add(f"A2  GOAL-ONLY 16 QUERY | Stage {stage} | {phase.upper()} | {datetime.now():%m-%d %H:%M:%S}", 1)
    if stage in ('I', 'II'):
        add(f"{plan['world_size']} 进程同步训练：全局步数相同；各进程的微批次与推进状态分别显示。")
    else:
        add(f"{data['evaluation_settings'].get('workers', 6)} 个推理进程并行评测；模型逐个加载，任务自动分配。")
    a = data['states']['I'].get('step', 0)
    b = data['states']['II'].get('step', 0)
    add(f"I {bar(a,1000,10)} {a:4}/1000  II {bar(b,2000,10)} {b:4}/2000  Eval {data['episodes']:4}/2000", 2)
    metric = data['metric']
    if stage in ('I', 'II'):
        target = plan[f'stage_{stage}']['steps']
        completed = state.get('step', 0)
        remaining = (target-completed)*data['average'] if data['average'] else None
        add(f"Loss {number(metric.get('loss'))} | Flow {number(metric.get('flow'))} | GoalKD {number(metric.get('goal_distill'))} | Grad {number(metric.get('grad_norm'), '.3f')}")
        add(f"Avg {number(data['average'], '.1f')} s/step | elapsed {duration(state.get('elapsed_seconds'))} | ETA ~{duration(remaining)}")
        last_update_age = now-metric['time'] if metric.get('time') else None
        add(f"近 5 分钟完成 {data['recent_steps']} 步 | 最近更新 {duration(last_update_age)} 前 | 停滞阈值 {duration(max(300., 3*(data['average'] or 0)))}")
    else:
        add(f"Full LIBERO: {data['tasks']}/40 tasks | {data['episodes']}/2000 episodes | successes {data['successes']}")
        add(f"Evaluation: Student step 2000 on GPU {plan['evaluation']['physical_gpu']}.")
    add()
    if stage not in ('I', 'II'):
        status = data.get('evaluation_status', {})
        lanes = status.get('workers', [])
        for lane in lanes:
            pid = lane.get('server_pid')
            live = alive(pid)
            lane_phase = lane.get('phase', 'waiting')
            age = lane.get('movement_age', 0)
            if lane_phase == 'evaluating':
                activity = '持续推进' if live and age < 60 else ('等待更新' if live and age < 300 else '需检查')
            else:
                activity = dict(queued='排队', waiting_for_load='等待加载', loading_model='加载模型',
                    ready='模型就绪', preparing='准备任务', retry_pending='重试等待', idle='任务结束').get(lane_phase, lane_phase)
            count = lane.get('episodes_completed', 0)
            gpu = lane.get('physical_gpu', plan['evaluation']['physical_gpu'])
            task = f"{lane.get('suite', '--')}/{lane.get('task_id', '--')}"
            color = 3 if live else (2 if lane_phase == 'idle' else 4)
            add(f"GPU {gpu} / worker {lane['slot']}  {bar(count,50,14)} {count:2}/50  {task}  PID {pid or '--'}", color)
            add(f"  {activity} | 推理调用 {lane.get('calls',0)} | 最近变化 {duration(age)} 前 | VRAM {number(stats.get('processes',{}).get(pid),'.2f')} GiB", color)
        if not lanes:
            add(f"等待六路评测状态：{status.get('phase', '正在校验最终权重')}", 4)
        if status.get('error') or phase == 'failed':
            error = status.get('error') or data['pipeline'].get('error', 'Evaluation failed')
            add(error.strip().splitlines()[-1], 5)
        add()
        add('Refresh 1s | GPU stats 10s | Ctrl-b 0: dashboard | Ctrl-b 1: logs', 4)
        add('Ctrl-b d: detach | q: close dashboard', 4)
        return content
    for item in data['workers']:
        gpu, rank = item['gpu'], item['rank']
        heartbeat, worker = item['heartbeat'], item['worker']
        pid = worker.get('pid')
        board = stats.get('boards', {}).get(gpu, {})
        if stage in ('I', 'II'):
            completed = state.get('step', 0)
            target = plan[f'stage_{stage}']['steps']
            is_alive = item['process']['alive']
            current_step = heartbeat.get('step', 0)
            micro = heartbeat.get('micro', 0) if current_step > completed else 0
            micro = min(micro, item['micro_total'])
            age = max(0, now-heartbeat['time']) if heartbeat.get('time') else None
            activity = state.get('phase', 'initializing')
            if activity == 'training':
                activity = 'sync / optimizer' if micro == item['micro_total'] else ('backward' if micro else 'loading next batch')
            health = 'LIVE' if is_alive else ('DONE' if state.get('phase') == 'completed' else 'WAIT')
            progress = movement(item, data, now)
            add(f"GPU {gpu} / rank {rank} {bar(completed,target,18)} {completed:4}/{target} {100*completed/target:5.1f}% PID {pid or '--'} {health}", progress['color'])
            add(f"  {progress['label']} | 最近推进 {duration(progress['age'])} 前 | CPU {number(item['process'].get('cpu_percent'),'.0f')}% ({item['process']['state']})", progress['color'])
            add(f"  update {max(completed+1, current_step) if completed < target else target} | micro {bar(micro,item['micro_total'],10)} {micro:2}/{item['micro_total']} | heartbeat {duration(age)}")
        else:
            pid = data['server'].get('pid') if gpu == plan['evaluation']['physical_gpu'] else None
            if pid:
                add(f"GPU {gpu} / inference  {bar(data['episodes'],2000,24)} {data['episodes']:4}/2000 episodes  PID {pid}", 3)
            else: add(f"GPU {gpu} / rank {rank}  {bar(2000,2000,24)} training complete", 2)
            add(f"  {'Inference running' if pid and alive(pid) else 'Training workers finished'}")
        job_mem = stats.get('processes', {}).get(pid)
        add(f"  VRAM A2 {number(job_mem,'.2f')} GiB | GPU total {number(board.get('used'),'.1f')}/{number(board.get('total'),'.0f')} GiB | GPU util {board.get('util','--')}%")
        add()
    if phase == 'failed':
        error = data['pipeline'].get('error', '').strip().splitlines()
        add(error[-1] if error else 'Pipeline failed: inspect train-logs window.', 5)
    cache_age = now-stats['time'] if stats.get('time') else None
    add(f"Refresh 1s | GPU stats 10s (age {duration(cache_age)}) | Ctrl-b 1: logs", 4)
    add('Ctrl-b d: detach | Ctrl-b 0: dashboard | q: close dashboard', 4)
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
    sampler = ProcessSampler()
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
            try:
                data = snapshot()
                sampler.update(data)
                rendered = lines(data, stats, width)
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
