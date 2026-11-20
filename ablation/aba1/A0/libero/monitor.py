#!/usr/bin/env python3
"""Progress display and persistent health log; never starts or controls training."""
import argparse,datetime,json,os,statistics,subprocess,time
from pathlib import Path

WORK=Path(__file__).resolve().parent

def read_json(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}

def duration(seconds):
    if seconds is None:return '等待更新速度'
    seconds=max(0,int(seconds));h,seconds=divmod(seconds,3600);m,s=divmod(seconds,60)
    return f'{h:02d}:{m:02d}:{s:02d}'

def gpu_rows(run,selected):
    owners=set()
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if str(run).encode() in args and b'train.py' in args and b'torch.distributed.run' not in args:
                owners.add(p.name)
        except OSError:pass
    query=lambda args:subprocess.check_output(['nvidia-smi',*args,'--format=csv,noheader,nounits'],text=True,timeout=5)
    devices=query(['--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu'])
    apps=query(['--query-compute-apps=gpu_uuid,pid,used_memory'])
    apps=[x.split(', ') for x in apps.strip().splitlines()]
    rows=[]
    for line in devices.splitlines():
        idx,uuid,used,total,util=line.split(', ')
        if int(idx) not in selected:continue
        mine=[(pid,memory) for gpu,pid,memory in apps if gpu==uuid and pid in owners]
        details=', '.join(f'PID {pid}: {memory} MiB' for pid,memory in mine) or '暂无训练进程'
        rows.append(f'GPU {idx}  本任务 {details} | 全卡 {used}/{total} MiB | 利用率 {util}%')
    return rows,len(owners)

def render(run,gpus):
    cfg=read_json(run/'config.json') or read_json(WORK/'config.json')
    state=read_json(run/'status.json');pipe=read_json(run/'pipeline_status.json')
    rows=[]
    try:
        for line in (run/'metrics.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except ValueError:pass
    except OSError:pass
    train=[r for r in rows if r.get('split')=='train'];last=train[-1] if train else {}
    total=cfg.get('total_steps',500);step=max(state.get('step',0),last.get('step',0))
    samples=max(state.get('seen_anchors',0),last.get('seen_anchors',0))
    fraction=min(1,step/total);filled=int(40*fraction)
    recent=train[-20:];speed=statistics.median(r['update_seconds'] for r in recent) if recent else None
    phase=state.get('phase','等待启动');pipeline=pipe.get('phase','等待启动')
    if phase=='completed' and pipeline in ('quick','full'):phase='训练完成，正在进行 '+pipeline+' 推理'
    labels={'loading':'加载权重','training':'训练','validation':'离线验证','checkpointing':'保存检查点','completed':'训练完成','failed':'失败，请查看日志'}
    gpu_label=','.join(map(str,cfg['physical_gpus']))
    lines=[f'A0 LIBERO | 单组 latent16 / 无 DINO | GPU {gpu_label} | {datetime.datetime.now():%Y-%m-%d %H:%M:%S}',
        f'运行：{run.name}',f'状态：{labels.get(phase,phase)}   pipeline={pipeline}',
        f"[{'#'*filled}{'-'*(40-filled)}] {step:3d}/{total} steps  {fraction:6.2%}",
        f'样本访问：{samples:,}/{total*32:,}   warmup：{min(step,cfg["warmup_steps"])}/{cfg["warmup_steps"]} steps',
        f'等效数据 epoch：{samples/260726:.5f}   有效全局 batch=32',
        f'近期速度：{speed:.2f} 秒/更新   剩余更新估时：{duration((total-step)*speed)}' if speed else '近期速度：等待首个完整参数更新',
        '估时不含后续验证、保存和 LIBERO 推理；共享资源可能使速度变化。']
    if last:
        age=time.time()-last['time']
        lines += [f'最近 loss={last["loss"]:.7f}   grad_norm={last["grad_norm"]:.5f}   lr={last["lr"]:.3e}',
                  f'距最近更新 {duration(age)}；最近更新耗时 {last["update_seconds"]:.2f} 秒',
                  f'训练张量峰值 {last.get("max_rank_peak_gib",0):.2f} GiB；缓存预留 {last.get("max_rank_reserved_gib",0):.2f} GiB']
    lines+=['',*gpus]
    latest=read_json(run/'latest.json')
    lines += ['',f'最近完整检查点：step {latest["step"]} | {Path(latest["path"]).name}' if latest else '最近完整检查点：尚未保存（每100步，以及250/500步保存）',
              f'训练日志：{run}/training.log',
              '此窗口只读监控。Ctrl-b 然后 d 脱离 tmux；训练继续运行。']
    if state.get('error') or pipe.get('error'):lines.append('错误：'+str(state.get('error') or pipe.get('error')))
    lines += ['', '自动流程：step500 完整保存 → quick 400 episodes → full 2000 episodes']
    for protocol,folder,expected in [('quick','quick_40x10',400),('full','full_40x50',2000)]:
        out=WORK/'evaluation'/run.name/folder;summary=read_json(out/'summary.json');evaluation_state=read_json(out/'status.json')
        if summary or pipeline==protocol:
            done=summary.get('episodes_completed',0);k=int(30*min(1,done/expected))
            lines.append(f"{protocol} [{'#'*k}{'-'*(30-k)}] {done}/{expected} episodes（按完成任务汇总）")
            if summary.get('success_rate') is not None:lines.append(f"已汇总成功率：{summary['success_rate']:.2%}")
            if evaluation_state.get('phase')=='evaluating':lines.append(f"正在评估：{evaluation_state['suite']} / task {evaluation_state['task_id']}")
            if evaluation_state.get('error'):lines.append('评估错误：'+evaluation_state['error'])
    return '\n'.join(lines)

def write_health(run,last_key):
    state=read_json(run/'status.json');pipe=read_json(run/'pipeline_status.json');launch=read_json(run/'launch.json')
    pid=pipe.get('pid',launch.get('pid'));alive=False
    if pid:
        try:alive=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[0]!='Z'
        except OSError:pass
    phase=pipe.get('phase','unknown');age=time.time()-state.get('time',time.time());warning=None
    if phase=='failed' or state.get('phase')=='failed':warning='后台流程失败，请查看日志；不会使用未完成权重评估'
    elif phase!='completed' and not alive:warning='后台流程进程未运行，请检查 pipeline.log'
    elif phase=='training' and state.get('phase')=='training' and age>1800:warning='超过30分钟无完整训练更新，请检查资源和训练日志'
    full=read_json(WORK/'evaluation'/run.name/'full_40x50/summary.json')
    value=dict(time=time.time(),monitor_pid=os.getpid(),pipeline_pid=pid,pipeline_alive=alive,
               pipeline_phase=phase,training_phase=state.get('phase'),step=state.get('step',0),
               total_steps=500,full_episodes=full.get('episodes_completed',0),full_expected_episodes=2000,
               warning=warning,automatic_evaluation=['quick','full'])
    target=run/'monitor_heartbeat.json';temp=target.with_name(target.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(target)
    key=(value['step'],phase,value['training_phase'],value['full_episodes'],warning)
    if key!=last_key:
        with (run/'monitor_events.jsonl').open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
    return key,warning

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--interval',type=float,default=5);p.add_argument('--once',action='store_true');args=p.parse_args()
    run=Path(args.run).resolve()
    if not run.is_relative_to(WORK/'runs'):raise ValueError('Monitor run must be in A0/libero/runs')
    last_key=None
    while True:
        selected=(read_json(run/'config.json') or read_json(WORK/'config.json'))['physical_gpus']
        try:gpus,count=gpu_rows(run,selected);gpus.insert(0,f'训练 worker：{count}/3（每卡一个训练 rank）')
        except (OSError,subprocess.SubprocessError,ValueError) as e:gpus=['GPU 查询暂不可用：'+str(e)]
        last_key,warning=write_health(run,last_key)
        text=render(run,gpus)
        if warning:text+='\n监控提示：'+warning
        print(('' if args.once else '\033[2J\033[H')+text,flush=True)
        if args.once:return
        time.sleep(max(1,args.interval))

if __name__=='__main__':main()
