"""Persistent tmux dashboard; reads atomic status and append-only training metrics."""
import argparse,json,time,math
from pathlib import Path
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress,BarColumn,TextColumn,TaskProgressColumn
from rich.console import Group


def read(path,default):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def duration(value):
    if value is None or not math.isfinite(value):return '--'
    seconds=max(0,int(value));return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'

def spark(values):
    if not values:return '--'
    lo,hi=min(values),max(values);chars='▁▂▃▄▅▆▇█'
    return ''.join(chars[min(7,int((v-lo)/max(1e-12,hi-lo)*7))] for v in values)

def render(root):
    status=read(root/'status.json',{});launch=read(root/'launch.json',{})
    stage=str(launch.get('stage','I'))
    preflight = None
    if status.get('phase')=='preflight' and launch.get('preflight_path'):
        preflight=read(Path(launch['preflight_path'])/'status.json',{})
        if preflight:
            status['phase']='预检：'+preflight.get('phase','loading')
            status['updated_at']=preflight.get('updated_at',status.get('updated_at'))
            if preflight.get('error'):status['error']=preflight['error']
    records=[]
    try:
        for line in (root/'metrics.jsonl').read_text().splitlines():
            try:records.append(json.loads(line))
            except ValueError:pass
    except OSError:pass
    train=[r for r in records if r.get('split')=='train'];val=[r for r in records if r.get('split')=='validation']
    step=int(status.get('step',train[-1]['step'] if train else 0));total=int(status.get('total_steps',launch.get('steps',10000)))
    progress=Progress(TextColumn('Stage '+stage),BarColumn(bar_width=48),TaskProgressColumn(),TextColumn('{task.completed:.0f}/{task.total:.0f} updates'))
    progress.add_task('train',total=total,completed=step)
    table=Table(show_header=False,expand=True);table.add_column('key',style='cyan');table.add_column('value')
    table.add_row('状态',status.get('phase','等待训练进程初始化'))
    table.add_row('GPU / batch',f"物理 GPU {launch.get('physical_gpu','--')} | micro={launch.get('batch_size','--')} × accumulation={launch.get('accumulation','--')}")
    elapsed=time.time()-status.get('started_at',time.time());table.add_row('已运行 / ETA',duration(elapsed)+' / '+duration(status.get('eta_seconds')))
    table.add_row('当前微步',f"{status.get('micro',0)}/{launch.get('accumulation','--')}")
    age=time.time()-status.get('updated_at',time.time());table.add_row('状态最后更新',f'{age:.0f} 秒前')
    if train:
        t=train[-1];recent=train[-100:];ema=recent[0]['flow']
        for r in recent[1:]:ema=.95*ema+.05*r['flow']
        table.add_row('Train action loss',f"{t['flow']:.6f} | EMA(0.95)={ema:.6f} | 最近100步均值={sum(r['flow'] for r in recent)/len(recent):.6f}")
        table.add_row('loss 趋势（最近60更新，自适应纵轴）',spark([r['flow'] for r in train[-60:]]))
        table.add_row('梯度范数 / LR',f"{t.get('grad_norm',0):.5g} / {t.get('lr',0):.3g}")
        table.add_row('最近更新耗时',duration(t.get('update_seconds')))
        if stage=='II':
            table.add_row('Train Local / Goal / Total',f"{t.get('local_distill',0):.6f} / {t.get('goal_distill',0):.6f} / {t['loss']:.6f}")
            table.add_row('Local 蒸馏趋势',spark([r['local_distill'] for r in train[-60:]]))
            table.add_row('Goal 蒸馏趋势',spark([r['goal_distill'] for r in train[-60:]]))
    if val:
        v=val[-1];table.add_row('固定验证 action loss',f"step {v['step']}: {v['flow']:.6f} | 初始={val[0]['flow']:.6f} | 最优={min(x['flow'] for x in val):.6f}")
        if 'without_roles_flow' in v:table.add_row('未来条件诊断（同噪声子集）',f"正常={v['diagnostic_flow']:.6f} | 关闭Goal/Local={v['without_roles_flow']:.6f}")
        if stage=='II':
            table.add_row('Validation Local / Goal / Total',f"{v.get('local_distill',0):.6f} / {v.get('goal_distill',0):.6f} / {v['loss']:.6f}")
    if stage=='II':
        audit=read(root/'teacher_freeze_audit.json',{})
        table.add_row('冻结 Teacher',f"Stage I step {launch.get('teacher_step','--')} | 冻结检查至学生 step {audit.get('checked_after_student_step','--')}")
    if len(val)>=2:
        change=(val[-1]['flow']/max(val[0]['flow'],1e-12)-1)*100
        table.add_row('验证 loss 相对初始',f'{change:+.2f}%（负数表示下降）')
        if len(val)>=5 and min(x['flow'] for x in val[-4:])>=min(x['flow'] for x in val[:-4]):
            table.add_row('趋势提示','最近 4 次验证未创新低，请复核学习率与未来条件利用')
    table.add_row('最近检查点',str(status.get('checkpoint','--')))
    if status.get('error'):table.add_row('错误',status['error'])
    table.add_row('日志',str(root/'train.log'));table.add_row('退出查看', 'Ctrl+b，然后 d（训练继续运行）')
    return Panel(Group(progress,table),title='LaWAM aba2_change · '+('学生蒸馏 Stage II' if stage=='II' else '后验教师 Stage I'),subtitle='2 秒刷新；action loss ≠ LIBERO 成功率')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--once',action='store_true');args=p.parse_args();root=Path(args.run)
    if args.once:Console().print(render(root));return
    with Live(render(root),refresh_per_second=.5,screen=True) as live:
        while True:time.sleep(2);live.update(render(root))
if __name__=='__main__':main()
