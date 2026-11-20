"""Compact, read-only terminal dashboard for the two nohup inference lanes."""
import argparse, datetime, shutil, time, unicodedata
from pathlib import Path
from eval_common import read

def width(s):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)
def fit(s,n):
    if width(s)<=n:return s
    out=''
    for c in s:
        if width(out+c)>n-1:break
        out+=c
    return out+'…'
def rate(row):
    n=row.get('episodes',0);s=row.get('successes',0)
    return f'{s}/{n} = {s/n:.2%}' if n else '暂无已完成episode'
def render(run):
    r=read(run/'status.json');size=shutil.get_terminal_size((90,20))
    n=r.get('episodes',0);cells=r.get('completed_cells',0);skip=r.get('skipped_cells',0)
    phases={'loading':'加载权重','evaluating':'推理中','ready':'已就绪','recovering':'恢复服务','pending':'等待启动'}
    lines=[f'RoboTwin Stage III 最终评测 | GPU5 双通道 | {datetime.datetime.now():%m-%d %H:%M:%S}',
           '冻结交互学生200步 + Actor100 | 50任务 × 2模式 × 50条',
           f'已完成任务模式 {cells}/100 | 跳过 {skip} | 剩余队列 {r.get("remaining_jobs",100)}',
           '['+'#'*int(28*n/5000)+'-'*(28-int(28*n/5000))+f'] {n}/5000 ({n/5000:.1%})',
           'Clean      '+rate(r.get('modes',{}).get('demo_clean',{})),
           'Randomized '+rate(r.get('modes',{}).get('demo_randomized',{})),
           '合计       '+rate(r)]
    for name in ('0','1'):
        lane=r.get('lanes',{}).get(name,{});a=lane.get('activity',{})
        lines.append(f'通道{name}：{phases.get(lane.get("phase"),lane.get("phase","等待"))}  '
                     f'{lane.get("task","-")} / {lane.get("mode","-").removeprefix("demo_")}')
        lines.append(f'  已完成{lane.get("episodes",0)}/50 成功{lane.get("successes",0)} | '
                     f'环境步{a.get("env_steps",0)} | {lane.get("no_progress_seconds",0)}秒无更新')
    lines += [f'总状态：{r.get("phase","准备中")} | 记录时间：'+
              (datetime.datetime.fromtimestamp(r['updated_at']).strftime('%H:%M:%S') if r.get('updated_at') else '-'),
              '40分钟无实际进展：跳过当前任务模式，继续队列',
              '成功率按已完成episode统计；跳过与缺失单独记录',
              '每5秒刷新 | Ctrl+b 然后 d 退出查看；nohup推理继续']
    if r.get('error'):lines.insert(2,'错误：'+r['error'])
    return '\n'.join(fit(x,max(1,size.columns-1)) for x in lines[:max(1,size.lines-1)])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--once',action='store_true')
    args=p.parse_args();run=Path(args.run)
    if args.once:print(render(run))
    else:
        while True:
            print('\033[2J\033[H'+render(run),end='',flush=True);time.sleep(5)
