"""Read-only, resize-aware live dashboard for training and all 20 quick evaluations."""
import argparse,curses,json,locale,math,os,time,unicodedata
from pathlib import Path
from .monitor_stage_two import read,metrics,plot

MODES=('demo_clean','demo_randomized')
KEYS=(('flow','Action'),('local_distill','Local'),('goal_distill','Goal'),('loss','Total'))

def cells(text):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in ('W','F') else 1 for c in str(text))

def truncate(text,width):
    out='';n=0
    for c in str(text):
        k=cells(c)
        if n+k>width:break
        out+=c;n+=k
    return out

def pad(text,width,right=False):
    value=truncate(str(text),width);space=' '*max(0,width-cells(value))
    return space+value if right else value+space

def num(value):
    return '--' if value is None or not math.isfinite(value) else f'{value:.6f}'

def rate(success,episodes):
    return f'{100*success/episodes:.1f}%' if episodes else '--'

def duration(seconds):
    if seconds is None or not math.isfinite(seconds):return '--'
    seconds=max(0,int(seconds));days,seconds=divmod(seconds,86400);h,seconds=divmod(seconds,3600);m=seconds//60
    return (f'{days}d ' if days else '')+f'{h:02d}h {m:02d}m'

def bar(value,total,width):
    ratio=min(1,max(0,value/max(1,total)));units=int(width*ratio*8);done,partial=divmod(units,8)
    edge='▏▎▍▌▋▊▉'[partial-1] if partial else ''
    return '['+'█'*done+edge+'░'*max(0,width-done-int(bool(partial)))+']'

def finite_values(rows,key):
    return [float(r[key]) for r in rows if key in r and math.isfinite(float(r[key]))]

def spark(values,width):
    if not values:return '等待首个训练更新'
    values=[math.log10(max(v,1e-12)) for v in values];low,high=min(values),max(values)
    chars='▁▂▃▄▅▆▇█';result=''
    for i in range(width):
        lo=i*len(values)//width;hi=max(lo+1,(i+1)*len(values)//width)
        value=sum(values[lo:hi])/len(values[lo:hi])
        result+=chars[min(7,int(7*(value-low)/max(high-low,1e-9)))]
    return result

def mode_state(run,step,mode,queue):
    base=run/'quick_eval';tag=f'step_{step:06d}'
    report_path=base/f'{tag}_{mode}.json';report=read(report_path,None)
    directory=base/f'{run.name}__{mode}'/tag
    schedule=read(directory/'.task_scheduler.json',{})
    active=queue.get('phase')=='evaluating' and queue.get('step')==step and queue.get('mode')==mode
    if report is not None:
        episodes=int(report.get('episodes',0));success=int(report.get('successes',0))
        skipped=int(report.get('skipped_tasks',0));completed=int(report.get('completed_tasks',0))
        errors=max(0,50-completed-skipped)
    else:
        rows=[read(p,{}) for p in (directory/'tasks').glob('*/progress.json')]
        episodes=sum(int(r.get('completed_episodes',0)) for r in rows)
        success=sum(int(r.get('successes',0)) for r in rows)
        completed=sum(int(r.get('completed_episodes',0))>=10 for r in rows)
        skipped_paths=list((directory/'tasks').glob('*/skipped_summary.json'))
        skipped=len(skipped_paths);skipped_names={p.parent.name for p in skipped_paths}
        errors=len(set(schedule.get('failed',{}))-skipped_names)
    tasks=[]
    for task,details in schedule.get('in_progress',{}).items():
        progress=read(directory/'tasks'/task/'progress.json',{})
        worker=int(details.get('worker_index',0));gpu=(0,5)[min(worker,1)]
        tasks.append(dict(task=task,gpu=gpu,episodes=int(progress.get('completed_episodes',0)),successes=int(progress.get('successes',0))))
    return dict(episodes=episodes,successes=success,completed=completed,skipped=skipped,errors=errors,
        final=report is not None,active=active,tasks=tasks)

def snapshot(run):
    status=read(run/'status.json',{});rows=metrics(run/'metrics.jsonl')
    # Resume history can contain duplicate step entries; show the most recent row.
    trains={r['step']:r for r in rows if r.get('split')=='train'}
    vals={r['step']:r for r in rows if r.get('split')=='validation'}
    train=[trains[k] for k in sorted(trains)];validation=[vals[k] for k in sorted(vals)]
    queue=read(run/'quick_eval/status.json',{})
    cancelled=set(read(run/'quick_eval/cancelled_steps.json',{}).get('steps',[]))
    total=int(status.get('total_steps',10000));checkpoints=[]
    for step in range(500,total+1,500):
        modes=[mode_state(run,step,m,queue) for m in MODES]
        saved=(run/f'step_{step:06d}/checkpoint_complete.json').exists()
        episodes=sum(m['episodes'] for m in modes);successes=sum(m['successes'] for m in modes)
        skipped=sum(m['skipped'] for m in modes);errors=sum(m['errors'] for m in modes)
        finished=all(m['final'] for m in modes)
        if step in cancelled:phase='已取消'
        elif finished:phase='有缺项' if errors else '完成/跳过' if skipped else '已完成'
        elif any(m['active'] for m in modes):phase='评测中'
        elif any(m['final'] or m['episodes'] for m in modes):phase='部分完成'
        elif saved:phase='排队中'
        else:phase='待保存'
        checkpoints.append(dict(step=step,modes=modes,saved=saved,episodes=episodes,successes=successes,
            skipped=skipped,errors=errors,finished=finished,cancelled=step in cancelled,phase=phase))
    pid=status.get('pid');alive=False
    if pid:
        try:os.kill(int(pid),0);alive=True
        except OSError:pass
    return dict(status=status,train=train,validation=validation,queue=queue,checkpoints=checkpoints,alive=alive)

def training_lines(data,width):
    s=data['status'];train=data['train'];step=max(int(s.get('step',0)),train[-1]['step'] if train else 0)
    total=int(s.get('total_steps',10000));phase=s.get('phase','初始化')
    names={'training':'训练中','optimizer_step':'优化器更新','saving_checkpoint':'保存检查点','saving_best':'保存最佳模型','completed':'已完成','failed':'已停止/异常'}
    phase=names.get(phase,phase);suffix=f' {step:,}/{total:,}  {100*step/max(total,1):.2f}%'
    speed=sum(r['update_seconds'] for r in train[-20:])/len(train[-20:]) if train else None
    alive='进程存活' if data['alive'] else '进程已退出'
    age=max(0,time.time()-float(s.get('updated_at',time.time())))
    result=[f'训练 GPU2,5  |  {bar(step,total,max(10,width-cells(suffix)-18))}{suffix}',
        f'{phase}  微步 {s.get("micro",0)}/16  |  {speed:.1f}s/步  ETA {duration(s.get("eta_seconds"))}' if speed else f'{phase}  等待首步',
        f'{alive} | 状态更新 {age:.0f}s 前 | 已跳过异常更新 {s.get("skipped_updates",0)} | 教师 Stage I / 1000 冻结']
    if s.get('error'):result[-1]='错误: '+s['error']
    return result

def loss_table(data,width):
    train=data['train'];val=data['validation'];latest=train[-1] if train else {};v=val[-1] if val else {}
    val_label=f'验证(step {v.get("step","--")})'
    lines=[f'损失             最新训练       最近50步均值    {val_label:<16} 首/末窗口变化']
    for key,label in KEYS:
        values=finite_values(train,key);tail=values[-50:];head=values[:min(50,len(values))]
        avg=sum(tail)/len(tail) if tail else None
        change=100*(avg/(sum(head)/len(head))-1) if head and sum(head)>0 else None
        diff=f'{change:+.1f}%' if change is not None and len(values)>50 else '--'
        lines.append(f'{label:<10} {num(latest.get(key)):>14} {num(avg):>18} {num(v.get(key)):>14} {diff:>15}')
    return lines

def eval_header(data,width):
    checkpoints=data['checkpoints'];queue=data['queue'];active=next((r for r in checkpoints if any(m['active'] for m in r['modes'])),None)
    saved=sum(r['saved'] for r in checkpoints);finished=sum(r['finished'] for r in checkpoints)
    cancelled=sum(r.get('cancelled',False) for r in checkpoints)
    waiting=sum(r['saved'] and not r['finished'] and not r.get('cancelled',False) for r in checkpoints)-int(active is not None)
    if active:
        suffix=f' {active["episodes"]}/1000  成功 {rate(active["successes"],active["episodes"])}'
        current=f'快测 step {active["step"]} '+bar(active['episodes'],1000,max(10,width-cells(suffix)-20))+suffix
    elif saved:
        current='快测 GPU0,5：'+('全部已处理' if finished+cancelled==len(checkpoints) else '等待调度 / 下一个检查点')
    else:current='快测 GPU0,5：等待 step 500 权重；目前尚无正式快测成功率'
    second=f'权重 {saved}/{len(checkpoints)} | 已评测 {finished}/{len(checkpoints)} | 取消 {cancelled} | 排队 {max(0,waiting)} | 每权重 50任务×2模式×10次 | 无进度40分钟跳过'
    return [current,second]

def eval_table(rows,width):
    if width>=96:
        sizes=(7,12,12,20,20,12,9)
        header=('步数','状态','EP/1000','Clean EP / 成功率','Random EP / 成功率','总成功率','跳/错')
        def values(r):
            c,m=r['modes']
            return (str(r['step']),r['phase'],f'{r["episodes"]}/1000',f'{c["episodes"]}/500  {rate(c["successes"],c["episodes"])}',f'{m["episodes"]}/500  {rate(m["successes"],m["episodes"])}',rate(r['successes'],r['episodes']),f'{r["skipped"]}/{r["errors"]}')
    else:
        sizes=(6,11,11,12,12,12,7)
        header=('步数','状态','EP/1000','Clean%','Random%','总成功率','跳/错')
        def values(r):
            c,m=r['modes'];return (str(r['step']),r['phase'],f'{r["episodes"]}/1000',rate(c['successes'],c['episodes']),rate(m['successes'],m['episodes']),rate(r['successes'],r['episodes']),f'{r["skipped"]}/{r["errors"]}')
    line=lambda values:''.join(pad(x,w) for x,w in zip(values,sizes))
    return [line(header)]+[line(values(r)) for r in rows]

def chart(train,validation,key,label,width,height):
    if not train:return [label+'：等待数据']+['']*(height-1)
    values=finite_values(train,key)
    pairs=[(r['step'],float(r[key])) for r in train if key in r and math.isfinite(float(r[key]))]
    smooth=[];window=[]
    for step,value in pairs:
        window.append(value);window=window[-50:];smooth.append((step,sum(window)/len(window)))
    val=[(r['step'],float(r[key])) for r in validation if key in r and math.isfinite(float(r[key]))]
    logs=[math.log10(max(y,1e-12)) for x,y in pairs+smooth+val]
    lo,hi=min(logs),max(logs)
    if hi-lo<.01:hi+=.05;lo-=.05
    w=max(10,width-12);h=max(2,height-3);grid=[[' ']*w for _ in range(h)];last=max(1,pairs[-1][0])
    for series,symbol in [(pairs,'.'),(smooth,'*'),(val,'o')]:
        for x,y in series:
            col=min(w-1,round(x/last*(w-1)));row=min(h-1,max(0,round((hi-math.log10(max(y,1e-12)))/(hi-lo)*(h-1))))
            grid[row][col]=symbol
    lines=[f'{label}  当前 {values[-1]:.6f}  近50步 {smooth[-1][1]:.6f}  [log10纵轴]']
    for i,row in enumerate(grid):
        tick=10**(hi-i/max(1,h-1)*(hi-lo));lines.append(f'{tick:9.2g} |'+''.join(row))
    lines.append('          +'+('-'*w));lines.append('           0'+(' '*max(0,w-len(str(last))-1))+str(last))
    return lines

def render(data,width,height,view='overview',offset=0):
    lines=[('RoboTwin Stage II  |  '+time.strftime('%m-%d %H:%M:%S')+'  |  每2秒刷新',1)]
    lines.extend((s,2 if i==0 else 0) for i,s in enumerate(training_lines(data,width)))
    if view=='loss':
        lines.append(('完整训练历史：. 原始 loss   * 最近50步均值   o 固定验证值；横轴=学生训练步',1))
        panel=max(6,(height-7)//3)
        for key,label in KEYS[:3]:lines.extend((s,0) for s in chart(data['train'],data['validation'],key,label,width,panel))
    else:
        if view=='overview':
            lines.append(('─'*width,1));lines.extend((s,1 if i==0 else 0) for i,s in enumerate(loss_table(data,width)))
            for key,label in KEYS[:3]:
                lines.append((f'{label:<7} '+spark(finite_values(data['train'],key),max(10,width-10)),3))
            lines.append(('上面三条趋势覆盖全部训练步，各自按 log loss 缩放；按 2 查看带坐标曲线',0))
        lines.extend((s,1) for s in eval_header(data,width))
        if view=='overview':
            tasks=[t for r in data['checkpoints'] for m in r['modes'] if m['active'] for t in m['tasks']]
            lines.append(('当前任务：'+' | '.join(f'GPU{t["gpu"]} {t["task"]} {t["episodes"]}/10' for t in tasks) if tasks else '当前任务：等待正式快测；首步 smoke 结果不计入下表',0))
        available=max(1,height-len(lines)-3);n=len(data['checkpoints']);offset=max(0,min(offset,max(0,n-available)))
        selected=data['checkpoints'][offset:offset+available]
        lines.extend((s,1 if i==0 else 2 if selected[i-1]['phase']=='评测中' else 0) for i,s in enumerate(eval_table(selected,width)))
        if offset or len(selected)<n:lines.append((f'检查点 {offset+1}–{offset+len(selected)}/{n}  PgUp/PgDn 翻页；按 3 展开全部检查点',0))
        else:lines.append(('成功率=成功/已完成episode；跳过/缺项单列，未评测显示 --',0))
    footer='1 总览  2 损失曲线  3 快测结果  PgUp/PgDn 翻页 | tmux退出但保持后台：Ctrl-b d'
    return lines[:max(1,height-1)]+[('',0)]*max(0,height-1-len(lines))+[(footer,1)]

def draw(screen,run,view):
    curses.curs_set(0);screen.timeout(2000)
    try:
        curses.start_color();curses.use_default_colors()
        for n,col in [(1,curses.COLOR_CYAN),(2,curses.COLOR_GREEN),(3,curses.COLOR_YELLOW)]:curses.init_pair(n,col,-1)
    except curses.error:pass
    offset=0
    while True:
        h,w=screen.getmaxyx();data=snapshot(run);screen.erase()
        for i,(line,style) in enumerate(render(data,max(20,w-1),h,view,offset)):
            try:screen.addstr(i,0,truncate(line,max(1,w-1)),curses.color_pair(style)|(curses.A_BOLD if style==1 else 0))
            except curses.error:pass
        screen.refresh();key=screen.getch()
        if key in (ord('1'),ord('2'),ord('3')):view={ord('1'):'overview',ord('2'):'loss',ord('3'):'eval'}[key];offset=0
        elif key in (curses.KEY_NPAGE,curses.KEY_DOWN):offset=min(19,offset+(5 if key==curses.KEY_NPAGE else 1))
        elif key in (curses.KEY_PPAGE,curses.KEY_UP):offset=max(0,offset-(5 if key==curses.KEY_PPAGE else 1))
        elif key in (ord('q'),27):view='overview';offset=0

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--view',choices=('overview','loss','eval'),default='overview')
    p.add_argument('--export-only',action='store_true');p.add_argument('--once',action='store_true')
    a=p.parse_args();locale.setlocale(locale.LC_ALL,'')
    if a.export_only:
        while True:
            try:plot(a.run,metrics(a.run/'metrics.jsonl'))
            except Exception as exc:print(f'Chart export error: {exc}',flush=True)
            time.sleep(60)
    elif a.once:
        print('\n'.join(line for line,style in render(snapshot(a.run),120,42,a.view)))
    else:curses.wrapper(draw,a.run,a.view)

if __name__=='__main__':main()
