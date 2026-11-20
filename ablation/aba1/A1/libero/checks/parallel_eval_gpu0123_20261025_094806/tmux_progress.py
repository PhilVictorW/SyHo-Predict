#!/usr/bin/env python3
"""Read-only tmux progress and liveness display for a saved A1 run."""
import argparse
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
from zoneinfo import ZoneInfo

from monitor import read, rows, duration, PHASE, WORK


def process(pid, run=None, boot_id=None):
    if not pid:
        return dict(alive=False, label='尚未启动', cpu=None)
    if boot_id and boot_id != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        return dict(alive=False, label='服务器已重启', cpu=None)
    try:
        folder = Path('/proc') / str(pid)
        stat = (folder/'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] in ('Z', 'X'):
            return dict(alive=False, label='已退出', cpu=None)
        command = (folder/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        if run and str(run) not in command:
            return dict(alive=False, label='PID 已复用', cpu=None)
        cpu = (int(stat[11]) + int(stat[12])) / os.sysconf('SC_CLK_TCK')
        return dict(alive=True, label='存活', cpu=cpu)
    except (FileNotFoundError, ProcessLookupError):
        return dict(alive=False, label='已退出', cpu=None)
    except (OSError, ValueError, IndexError):
        return dict(alive=None, label='状态不可读', cpu=None)


def gpu_usage(devices):
    query = ['nvidia-smi', '-i', ','.join(map(str, devices))]
    result = {}
    try:
        raw = subprocess.check_output(query + [
            '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, timeout=3)
        uuid_index = {}
        for line in raw.splitlines():
            index, uuid, used, total, util = [x.strip() for x in line.split(',')]
            index = int(index)
            uuid_index[uuid] = index
            result[index] = dict(used=float(used)/1024, total=float(total)/1024,
                                 utilization=util, processes={})
        raw = subprocess.check_output(query + [
            '--query-compute-apps=gpu_uuid,pid,used_memory',
            '--format=csv,noheader,nounits'], text=True, timeout=3)
        for line in raw.splitlines():
            uuid, pid, memory = [x.strip() for x in line.split(',')]
            if uuid in uuid_index:
                try:
                    result[uuid_index[uuid]]['processes'][int(pid)] = float(memory)/1024
                except ValueError:
                    pass
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return result


def bar(done, total, width=24):
    total = max(1, total)
    filled = min(width, max(0, int(width*done/total)))
    return '[' + '#' * filled + '-' * (width-filled) + f'] {done}/{total} ({done/total:.1%})'


def age(seconds):
    return '未知' if seconds is None else duration(max(0, seconds))


def clip(line, width):
    result, count = [], 0
    for char in line:
        size = 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
        if count + size > width-1:
            return ''.join(result) + '…'
        result.append(char)
        count += size
    return ''.join(result)


class Dashboard:
    def __init__(self, root, interval=5):
        self.root, self.interval = root, interval
        self.plan = read(root/'run_plan.json') or read(root/'launch.json').get('plan', {})
        self.launch = read(WORK/'nohup'/root.name/'launch_record.json')
        self.active = None
        self.history = deque()
        self.previous_cpu = {}

    def snapshot(self):
        if not self.plan:
            self.plan = read(self.root/'run_plan.json') or read(self.root/'launch.json').get('plan', {})
            if not self.plan:
                return ['等待启动器写入运行配置，稍后自动刷新。']
        now = time.time()
        pipeline = read(self.root/'pipeline.json')
        evaluation = read(WORK/'auto_eval'/self.root.name/'status.json')
        active = pipeline.get('stage', 'II' if pipeline.get('phase') == 'completed' else 'I')
        if active != self.active:
            self.active = active
            self.history.clear()
            self.previous_cpu.clear()
        world = pipeline.get('world_size', self.plan.get('world_size', 3))
        devices = pipeline.get('physical_gpus', self.plan.get('physical_gpus', [self.plan['physical_gpu']]))
        stages = {}
        for stage in ('I', 'II'):
            folder = self.root/('stage_'+stage)
            state, metrics = read(folder/'status.json'), rows(folder/'metrics.jsonl')
            last = metrics[-1] if metrics else {}
            completed = max(int(state.get('step', 0)), int(last.get('step', 0)))
            stages[stage] = dict(state=state, metrics=metrics, last=last, completed=completed,
                                  target=self.plan['stage_'+stage]['steps'])
        current = stages[active]
        phase = current['state'].get('phase', 'loading')
        recent = [float(r['seconds']) for r in current['metrics'][-20:] if r.get('seconds', 0) > 0]
        speed = statistics.mean(recent) if recent else None
        stale_limit = max(180, 3*statistics.median(recent)) if recent else 600
        launcher_pid = self.launch.get('processes', {}).get('training', {}).get('pid')
        launcher = process(launcher_pid, boot_id=self.launch.get('boot_id'))
        workers, tokens = [], []
        for rank in range(world):
            folder = self.root/('stage_'+active)
            worker, beat = read(folder/f'worker_{rank}.json'), read(folder/f'heartbeat_{rank}.json')
            initialization = read(folder/f'initialization_{rank}.json')
            pid = worker.get('pid') or initialization.get('pid')
            proc = process(pid, self.root, self.launch.get('boot_id'))
            share = len(range(rank, self.plan['global_batch'], world))
            micro = int(beat.get('micro', 0))
            token = max(0, (int(beat.get('step', 1))-1)*share + micro) if beat else 0
            tokens.append(token)
            cpu_delta = None
            if proc['cpu'] is not None:
                previous = self.previous_cpu.get((active, rank, pid))
                cpu_delta = max(0, proc['cpu']-previous) if previous is not None else None
                self.previous_cpu[(active, rank, pid)] = proc['cpu']
            workers.append(dict(rank=rank, pid=pid, proc=proc, worker=worker, beat=beat,
                                share=share, micro=micro, cpu_delta=cpu_delta, initialization=initialization))
        self.history.append((now, current['completed'], tokens))
        while len(self.history) > 1 and self.history[1][0] < now-120:
            self.history.popleft()
        baseline = self.history[0]
        elapsed = now-baseline[0]
        step_delta = max(0, current['completed']-baseline[1])
        micro_deltas = [max(0, token-baseline[2][r]) for r, token in enumerate(tokens)]
        update_age = max(0, now-current['last']['time']) if 'time' in current['last'] else None
        latest_signal = max([current['last'].get('time', 0)] + [x['beat'].get('time', 0) for x in workers])
        signal_age = now-latest_signal if latest_signal else None
        exited = any(x['pid'] and x['proc']['alive'] is False for x in workers)
        loaded = sum(bool(x['initialization'].get('model_loaded') or x['worker']) for x in workers)
        loading = [str(devices[x['rank']] if len(devices)>1 else devices[0]) for x in workers if x['initialization'].get('phase') == 'loading_model']
        if pipeline.get('phase') == 'failed' or phase == 'failed':
            overall = '训练失败'
        elif pipeline.get('phase') == 'completed':
            overall = {'evaluating':'全量推理进行中', 'completed':'训练及全量推理已完成', 'verifying_step2000':'验证最终学生权重', 'evaluation_retry_pending':'全量推理等待重试', 'evaluation_failed':'全量推理失败', 'watcher_stopped_or_failed':'自动推理 watcher 已停止'}.get(evaluation.get('phase'), '两阶段训练已完成，等待自动推理')
        elif launcher['alive'] is False or exited:
            overall = '进程已退出，请检查日志'
        elif phase in ('validating', 'saving'):
            overall = PHASE[phase] + '（训练心跳暂时暂停）'
        elif signal_age is not None and signal_age > stale_limit:
            overall = '长时间无训练更新，请检查日志'
        elif step_delta > 0:
            overall = f'持续推进：观测窗口内完成 +{step_delta} 步'
        elif any(micro_deltas):
            overall = '持续推进：已观察到微批次变化'
        elif phase == 'loading':
            overall = f'错峰加载 {loaded}/{world} 就绪' + ('，正在加载 GPU '+','.join(loading) if loading else '，等待训练初始化')
        else:
            overall = '等待下一次推进确认'
        samples = gpu_usage(devices)
        stamp = datetime.now(ZoneInfo('Asia/Singapore')).strftime('%m-%d %H:%M:%S')
        lines = [
            f'A1 LIBERO | GPU {",".join(map(str,devices))} | {stamp} UTC+8',
            f'状态：{overall} | 启动器{launcher["label"]} | 每 {self.interval:g}s 刷新',
            '教师 I  ' + bar(stages['I']['completed'], stages['I']['target']),
            '学生 II ' + bar(stages['II']['completed'], stages['II']['target']),
            '全量推理 ' + bar(evaluation.get('episodes_completed',0),2000) + f' | {evaluation.get("tasks_completed",0)}/40 任务',
        ]
        last = current['last']
        if last:
            lines.append(f'{active} {PHASE.get(phase,phase)} | 最近完成更新 {age(update_age)} 前 | loss={last["loss"]:.6f} | grad={last["grad_norm"]:.4f}')
        else:
            lines.append(f'{active} {PHASE.get(phase,phase)} | 尚无已完成的参数更新')
        saved = read(self.root/('stage_'+active)/'latest.json')
        interval = self.plan['stage_'+active]['save_every']
        lines.append(f'权重：每 {interval} 步保存，保留最新 {self.plan["stage_"+active]["keep"]} 个 | 最近保存 {saved.get("step","尚无")} 步')
        lines.append(f'近 {len(recent)} 步 {speed:.1f}s/步 | 本阶段剩余约 {duration((current["target"]-current["completed"])*speed)}'
                     if speed else '速度及剩余时间：首次完成训练步后计算')
        compact = sys.stdout.isatty() and shutil.get_terminal_size((140,36)).lines < 30
        for item in workers:
            rank, beat, proc = item['rank'], item['beat'], item['proc']
            physical = item['worker'].get('physical_gpu', devices[rank] if len(devices)>1 else devices[0])
            gpu = samples.get(physical, {})
            memory = gpu.get('processes', {}).get(item['pid'])
            memory_text = f'{memory:.1f} GiB' if memory is not None else '待查询'
            heartbeat_age = max(0, now-beat['time']) if 'time' in beat else None
            if item['pid'] and proc['alive'] is False:
                label = proc['label']
            elif phase == 'loading':
                init = item['initialization']
                labels = dict(waiting_turn='等待错峰加载', loading_model='正在加载模型', waiting_for_peers='模型已加载，等待其他 rank')
                label = labels.get(init.get('phase'), '初始化/加载中')
                if loaded == world:
                    label = '所有模型已加载，准备同步训练'
            elif phase in ('validating', 'saving', 'completed'):
                label = PHASE[phase]
            elif phase == 'failed' or pipeline.get('phase') == 'failed':
                label = '失败'
            elif proc['alive'] is None:
                label = proc['label']
            elif max(beat.get('time', 0), last.get('time', 0)) < now-stale_limit:
                label = '长时间无更新，待检查'
            elif item['micro'] >= item['share']:
                label = '等待参数更新/同步'
            elif micro_deltas[rank] > 0:
                label = '正在推进'
            elif beat:
                label = '等待下一次心跳'
            else:
                label = '模型已加载，准备训练'
            lines.append(f'GPU {physical} | rank {rank} | PID {item["pid"] or "-"} {proc["label"]} | 本进程显存 {memory_text}')
            detail = f'  {active} step {beat.get("step","-")} ' + bar(item['micro'], item['share'], 8 if compact else 12) + f' | {label}'
            if compact:
                detail += f' | 心跳 {age(heartbeat_age)} +{micro_deltas[rank]}'
            lines.append(detail)
            cpu_text = f' | CPU +{item["cpu_delta"]:.1f}s' if item['cpu_delta'] is not None else ''
            if not compact:
                lines.append(f'  心跳 {age(heartbeat_age)} 前 | 窗口内 +{micro_deltas[rank]} 个样本'+cpu_text)
        for physical in devices:
            sample = samples.get(physical)
            lines.append(f'GPU {physical} 全卡：显存 {sample["used"]:.1f}/{sample["total"]:.0f} GiB | 利用率 {sample["utilization"]}%（含其他任务）'
                         if sample else f'GPU {physical}：状态暂不可读')
        lines.append(f'观测窗口 {min(120,int(elapsed))}s：完成 +{step_delta} 步 | 无更新提示阈值 {int(stale_limit)}s')
        if pipeline.get('phase') == 'failed':
            lines.append('错误：'+pipeline.get('error','请查看日志').strip().splitlines()[-1])
        else:
            lines.append('CPU 活动仅表示进程工作；训练推进以步数和样本变化确认。')
        lines.append('Ctrl+b 然后 d：离开 | 0：进度 | 1：训练日志 | 2：推理日志')
        return lines

    def render(self):
        lines = self.snapshot()
        if sys.stdout.isatty():
            size = shutil.get_terminal_size((120, 30))
            while len(lines) >= size.lines and any(x.startswith('GPU ') and '全卡' in x for x in lines):
                lines.pop(next(i for i,x in enumerate(lines) if x.startswith('GPU ') and '全卡' in x))
            if len(lines) >= size.lines:
                lines = [x for x in lines if not x.startswith('CPU 活动')]
            return '\n'.join(clip(line, max(20,size.columns-1)) for line in lines)
        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('interval must be at least one second')
    dashboard = Dashboard(args.run.resolve(), args.interval)
    tty = sys.stdout.isatty() and not args.once
    if tty:
        sys.stdout.write('\033[?1049h\033[?25l')
    try:
        while True:
            try:
                output = dashboard.render()
            except Exception as error:
                output = f'读取暂时失败：{error}\n{args.interval:g} 秒后重试。'
            if tty:
                sys.stdout.write('\033[H\033[2J')
            print(output, flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if tty:
            sys.stdout.write('\033[?25h\033[?1049l')
            sys.stdout.flush()


if __name__ == '__main__':
    main()
