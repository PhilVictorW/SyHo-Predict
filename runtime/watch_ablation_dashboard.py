"""Read-only live GPU/task progress dashboard for the LaWM inference ablation."""
from __future__ import annotations
import argparse
import csv
import getpass
import io
import json
import shlex
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'results/ablation/inference_without_lawm_and_queries/full_40x50_20260915'
NAMES = {'libero_10': 'Long', 'libero_object': 'Object', 'libero_goal': 'Goal', 'libero_spatial': 'Spatial'}
COLORS = {'RUNNING': 'green', 'LOADING': 'yellow', 'QUEUED': 'dim', 'DONE': 'cyan', 'STOPPED': 'red', 'ERROR': 'bold red'}


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def episodes(path):
    rows = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            row = json.loads(line)
            rows[(int(row['task_id']), int(row['episode_idx']))] = row
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return list(rows.values())


def bar(done, total, width=16):
    fill = min(width, int(width * done / max(total, 1)))
    value = Text('[')
    value.append('━' * fill, style='cyan' if done >= total else 'green')
    value.append('·' * (width - fill), style='bright_black')
    value.append(f'] {done}/{total} ({100*done/max(total,1):.1f}%)')
    return value


def rate(success, n):
    return f'{100*success/n:.1f}%' if n else '—'


def age(seconds):
    if seconds is None:
        return '—'
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        return f'{seconds//60}m{seconds%60:02d}s'
    return f'{seconds//3600}h{seconds%3600//60:02d}m'


def process_state(root):
    active, servers = {}, {}
    try:
        output = subprocess.run(['ps', '-ww', '-u', getpass.getuser(), '-o', 'pid=,args='], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.TimeoutExpired):
        return active, servers
    for line in output.splitlines():
        if str(root) not in line:
            continue
        try:
            pid, command = line.strip().split(None, 1)
            args = shlex.split(command)
        except ValueError:
            continue
        if 'deployment/model_server/server_policy.py' in args and '--port' in args:
            servers[int(args[args.index('--port')+1])] = int(pid)
        if '--args.task-suite-name' in args and any(x.endswith('/eval_libero.py') for x in args):
            active[args[args.index('--args.task-suite-name')+1]] = int(pid)
    return active, servers


def gpu_state():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4)
        return {int(row[0]): dict(util=row[1].strip(), used=int(row[2]), total=int(row[3])) for row in csv.reader(io.StringIO(result.stdout)) if len(row)==4}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}


class Dashboard:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.meta = read_json(run_dir/'run_meta.json', {})
        self.mapping = self.meta.get('suite_gpu_map', {'libero_10':0,'libero_object':2,'libero_goal':3,'libero_spatial':2})
        self.trials = int(self.meta.get('episodes_per_task', 50))
        self.suites = list(NAMES)
        self.task_names = {}
        baseline = Path(self.meta.get('baseline_run', ''))
        for suite in self.suites:
            for row in episodes(baseline/'suites'/suite/'episodes.checkpoint.jsonl'):
                self.task_names[(suite, int(row['task_id']))] = row['task_description']
        self.gpus = {}
        self.active, self.servers = {}, {}
        self.telemetry_time = 0

    def snapshot(self):
        now = time.time()
        if now-self.telemetry_time >= 5:
            self.gpus = gpu_state()
            self.active, self.servers = process_state(ROOT)
            self.telemetry_time = now
        suites = {}
        for index, suite in enumerate(self.suites):
            directory = self.run_dir/'suites'/suite
            path = directory/'episodes.checkpoint.jsonl'
            rows = episodes(path)
            tasks = defaultdict(list)
            for row in rows:
                tasks[int(row['task_id'])].append(row)
                self.task_names[(suite, int(row['task_id']))] = row['task_description']
            n = len(rows)
            k = sum(bool(row['success']) for row in rows)
            errors = sum(bool(row.get('error')) for row in rows)
            task_done = sum(len(tasks[t]) >= self.trials for t in range(10))
            complete = n == 10*self.trials and errors == 0 and (directory/'summary.json').exists()
            exit_code = None
            try:
                exit_code = int((directory/'exit_code.txt').read_text().strip())
            except (OSError, ValueError):
                pass
            # Prefer the port printed by the launcher (supports auto-selected ports).
            port = 21940+index
            try:
                for line in (directory/'launcher.log').read_text().splitlines():
                    if line.strip().startswith('port:'):
                        port = int(line.split(':',1)[1].strip())
                        break
            except (OSError, ValueError):
                pass
            if complete:
                status = 'DONE'
            elif errors or (exit_code is not None and exit_code != 0):
                status = 'ERROR'
            elif suite in self.active:
                status = 'RUNNING'
            elif port in self.servers:
                status = 'LOADING'
            elif directory.exists():
                status = 'STOPPED'
            else:
                status = 'QUEUED'
            current = next((t for t in range(10) if len(tasks[t]) < self.trials), None)
            last = now-path.stat().st_mtime if path.exists() else None
            suites[suite] = dict(n=n,k=k,errors=errors,tasks=tasks,task_done=task_done,status=status,current=current,last=last,pid=self.active.get(suite),port=port)
        return suites

    def render_queue(self, state, width, height):
        now=time.time()
        if now-self.telemetry_time>=5:
            self.gpus=gpu_state()
            self.telemetry_time=now
        task_data={}
        for key,task in state['tasks'].items():
            rows=episodes(Path(task['output'])/'episodes.checkpoint.jsonl')
            path=Path(task['output'])/'episodes.checkpoint.jsonl'
            task_data[key]=dict(n=len(rows),k=sum(bool(r['success']) for r in rows),
                errors=sum(bool(r.get('error')) for r in rows),last=now-path.stat().st_mtime if path.exists() else None)
        total=sum(d['n'] for d in task_data.values())
        success=sum(d['k'] for d in task_data.values())
        errors=sum(d['errors'] for d in task_data.values())
        done=sum(d['n']==50 and d['errors']==0 for d in task_data.values())
        pending=sum(t['status'] in ('pending','retry') for t in state['tasks'].values())
        running=sum(t['status']=='running' for t in state['tasks'].values())
        title=Text('LaWAM aba2 | SHARED TASK QUEUE | GPU 0 / 1 / 2 / 3 / 6 / 7',style='bold cyan')
        title.append('    '+time.strftime('%Y-%m-%d %H:%M:%S'),style='white')
        summary=Text.assemble(bar(total,2000,20),f'   Tasks {done}/40 | Active {running} | Waiting {pending} | OK {success}/{total} | SR {rate(success,total)} | Errors {errors}')
        table=Table(box=box.SIMPLE,expand=True)
        for name in ['GPU / util / memory*','State / current task','Task episodes','Task success / SR','Tasks finished','Last result']:
            table.add_column(name,no_wrap=True,overflow='ellipsis')
        compact=[]
        for gpu_str,info in sorted(state['gpus'].items(),key=lambda pair:int(pair[0])):
            gpu=int(gpu_str)
            key=info.get('task')
            task=state['tasks'].get(key,{})
            d=task_data.get(key,dict(n=0,k=0,errors=0,last=None))
            g=self.gpus.get(gpu)
            label=f"GPU {gpu}  {g['util']}%  {g['used']/1024:.1f}/{g['total']/1024:.0f}G" if g else f'GPU {gpu}'
            status=info['status'].upper()
            name=f"{NAMES.get(task.get('suite'),'')} T{task.get('task_id',0):02d}" if task else '—'
            completed=sum(t['status']=='done' and t.get('gpu')==gpu for t in state['tasks'].values())
            style='green' if status=='RUNNING' else 'yellow' if status in ('STARTING','LOADING') else 'red' if status=='ERROR' else 'cyan'
            table.add_row(label,Text(f'{status}  {name}',style=style),bar(d['n'],50,8) if task else Text('—'),f"{d['k']}/{d['n']}  {rate(d['k'],d['n'])}" if task else '—',str(completed),age(d['last']))
            content=Text.assemble(bar(d['n'],50,8),f"  OK {d['k']}/{d['n']}  SR {rate(d['k'],d['n'])}") if task else Text(status,style=style)
            compact.append(Panel(content,title=f'{label} | {status} {name}',border_style=style))
        parts=[title,summary,table] if width>=120 else [title,summary,*compact]
        if width>=120:
            overview=Table(box=box.SIMPLE,expand=True)
            for name in ['Suite','Active GPUs','Progress','Tasks','Success / SR']:
                overview.add_column(name,no_wrap=True)
            for suite in self.suites:
                keys=[key for key,t in state['tasks'].items() if t['suite']==suite]
                n=sum(task_data[key]['n'] for key in keys)
                k=sum(task_data[key]['k'] for key in keys)
                active=','.join(str(state['tasks'][key]['gpu']) for key in keys if state['tasks'][key]['status']=='running') or '—'
                finished=sum(task_data[key]['n']==50 for key in keys)
                overview.add_row(NAMES[suite],active,bar(n,500,12),f'{finished}/10',f'{k}/{n}  {rate(k,n)}')
            parts.append(overview)
        if width>=160 and height>=36:
            grid=Table.grid(expand=True,padding=(0,1))
            panels=[]
            for suite in self.suites:
                detail=Table(box=None,padding=(0,1),expand=True)
                for name in ['Task@GPU','Progress','OK / SR']:detail.add_column(name,no_wrap=True)
                for task_id in range(10):
                    key=f'{suite}__task{task_id:02d}'
                    task=state['tasks'][key]
                    d=task_data[key]
                    fill=min(5,d['n']//10)
                    progress=Text('[')
                    progress.append('━'*fill,style='green')
                    progress.append('·'*(5-fill),style='bright_black')
                    progress.append(f"] {d['n']}/50")
                    gpu=task.get('gpu')
                    label=('▶' if task['status']=='running' else ' ')+f'{task_id:02d}@'+(str(gpu) if gpu is not None else '—')
                    detail.add_row(Text(label,style='bold yellow' if task['status']=='running' else 'white'),progress,f"{d['k']}/{rate(d['k'],d['n'])}")
                panels.append(Panel(detail,title=NAMES[suite],border_style='cyan'))
            grid.add_row(*panels)
            parts.append(grid)
        elif width>=120:
            parts.append(Text('Use 160+ columns and 36+ rows to display all 40 task bars.',style='dim'))
        parts.append(Text('Each GPU finishes one task (50 trials), then takes the next task from the shared queue. Refresh: 3s.',style='dim'))
        parts.append(Text('* GPU memory/utilization includes other processes. SR uses completed trials. Detach: Ctrl-b then d.',style='dim'))
        if now-state.get('updated_at',now)>90 and state.get('status')=='running':
            parts.append(Text('Queue state has not updated for >90s; inspect queue/launcher.log.',style='yellow'))
        return Group(*parts)

    def render(self, width, height):
        queue_state=read_json(self.run_dir/'queue/state.json')
        if queue_state:
            return self.render_queue(queue_state,width,height)
        data = self.snapshot()
        total = sum(s['n'] for s in data.values())
        success = sum(s['k'] for s in data.values())
        errors = sum(s['errors'] for s in data.values())
        done_tasks = sum(s['task_done'] for s in data.values())
        title = Text('LaWAM aba2 | LIBERO | inference without LaWM and query outputs', style='bold cyan')
        title.append('    '+time.strftime('%Y-%m-%d %H:%M:%S'), style='white')
        summary = Text.assemble(bar(total, 40*self.trials, 24), f'   Tasks {done_tasks}/40   Success {success}/{total}   SR {rate(success,total)}   Errors {errors}')
        gpu_table = Table(box=box.SIMPLE, expand=True, padding=(0,1))
        for name in ['GPU / util / memory*','Suite / state','Assigned episodes','Success / SR','Current task / last result']:
            gpu_table.add_column(name, no_wrap=True, overflow='ellipsis')
        for gpu in sorted(set(self.mapping.values())):
            assigned = [s for s in self.suites if self.mapping[s] == gpu]
            focus = next((s for s in assigned if data[s]['status'] in ('RUNNING','LOADING','ERROR','STOPPED')), assigned[-1])
            info = data[focus]
            g = self.gpus.get(gpu)
            gpu_label = f"GPU {gpu}  {g['util']}%  {g['used']/1024:.1f}/{g['total']/1024:.0f}G" if g else f'GPU {gpu}  telemetry unavailable'
            states = Text()
            for i,suite in enumerate(assigned):
                if i: states.append(' → ')
                states.append(f"{NAMES[suite]} {data[suite]['status']}", style=COLORS[data[suite]['status']])
            n=sum(data[s]['n'] for s in assigned)
            k=sum(data[s]['k'] for s in assigned)
            task=info['current']
            current=f"T{task:02d} {len(info['tasks'][task])}/{self.trials}" if task is not None else 'all tasks complete'
            gpu_table.add_row(gpu_label,states,bar(n,len(assigned)*10*self.trials,8),f'{k}/{n}  {rate(k,n)}',f"{current}  |  {age(info['last'])} ago")
        overview = Table(box=box.SIMPLE, expand=True)
        for name in ['Suite','GPU','State','Episodes','Tasks','Success / SR','Active task description']:
            overview.add_column(name, no_wrap=True, overflow='ellipsis')
        for suite in self.suites:
            s=data[suite]
            task=s['current']
            description=self.task_names.get((suite,task),'—') if task is not None else 'complete'
            overview.add_row(NAMES[suite],str(self.mapping[suite]),Text(s['status'], style=COLORS[s['status']]),bar(s['n'],10*self.trials,8),f"{s['task_done']}/10",f"{s['k']}/{s['n']}  {rate(s['k'],s['n'])}",description)
        if width < 120:
            parts=[title, summary]
            for gpu in sorted(set(self.mapping.values())):
                assigned=[suite for suite in self.suites if self.mapping[suite]==gpu]
                n=sum(data[suite]['n'] for suite in assigned)
                k=sum(data[suite]['k'] for suite in assigned)
                g=self.gpus.get(gpu)
                telemetry=f"util {g['util']}% · memory {g['used']/1024:.1f}G" if g else 'telemetry unavailable'
                lines=[Text.assemble(bar(n,len(assigned)*10*self.trials,10),f'  OK {k}/{n}  SR {rate(k,n)}')]
                for suite in assigned:
                    info=data[suite]
                    task=info['current']
                    detail=f"T{task:02d} {len(info['tasks'][task])}/{self.trials}" if task is not None else 'complete'
                    lines.append(Text(f"{NAMES[suite]} {info['status']} · {detail} · SR {rate(info['k'],info['n'])} · last {age(info['last'])}",style=COLORS[info['status']]))
                parts.append(Panel(Group(*lines),title=f'GPU {gpu} | {telemetry}',border_style='cyan'))
        else:
            # Limit description width so the numeric columns remain visible.
            overview.columns[-1].max_width=max(20,width-100)
            parts=[title,summary,gpu_table,overview]
        if width >= 160 and height >= 35:
            grid=Table.grid(expand=True,padding=(0,1))
            panels=[]
            for suite in self.suites:
                table=Table(box=None,padding=(0,1),expand=True)
                table.add_column('Task',no_wrap=True)
                table.add_column('Progress',no_wrap=True)
                table.add_column('OK / SR',no_wrap=True)
                s=data[suite]
                for task in range(10):
                    rows=s['tasks'][task]
                    n=len(rows); k=sum(bool(x['success']) for x in rows)
                    active=task==s['current'] and s['status']=='RUNNING'
                    label=Text(('▶' if active else ' ')+f'{task:02d}', style='bold yellow' if active else 'white')
                    task_bar=Text('[')
                    task_bar.append('━'*min(5,5*n//self.trials),style='green')
                    task_bar.append('·'*(5-min(5,5*n//self.trials)),style='bright_black')
                    task_bar.append(f'] {n}/{self.trials}')
                    table.add_row(label,task_bar,f'{k} / {rate(k,n)}')
                panels.append(Panel(table,title=f'{NAMES[suite]} · GPU {self.mapping[suite]}',border_style='cyan'))
            grid.add_row(*panels)
            parts.append(grid)
        else:
            parts.append(Text('Widen terminal to at least 160 columns / 35 rows to show all 40 task bars.',style='dim'))
        parts.append(Text('* GPU memory/utilization includes other processes. SR uses completed episodes only. Refresh every 3s.',style='dim'))
        parts.append(Text('Detach: Ctrl-b then d | Progress persists after completion | Run: '+self.run_dir.name,style='dim'))
        return Group(*parts)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--refresh-sec',type=float,default=3)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    dashboard=Dashboard(args.run_dir)
    console=Console()
    if args.once:
        console.print(dashboard.render(console.width,console.height))
        return
    with Live(console=console, screen=True, auto_refresh=False) as live:
        while True:
            live.update(dashboard.render(console.width,console.height),refresh=True)
            time.sleep(max(1,args.refresh_sec))

if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
