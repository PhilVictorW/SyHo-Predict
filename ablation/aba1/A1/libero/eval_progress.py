#!/usr/bin/env python3
"""Live four-GPU A1 evaluation progress, independent of the evaluator."""
import argparse
import json
from pathlib import Path
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from tmux_progress import bar, process, gpu_usage, clip
from watch_full_eval import read
WORK=Path(__file__).resolve().parent
LABELS=dict(queued='等待加载',waiting_for_load='等待错峰加载',loading_model='加载模型',
    ready='模型就绪',preparing='准备任务',evaluating='推理中',retry_pending='等待重试',idle='任务已处理完')
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--interval',type=float,default=5)
    p.add_argument('--once',action='store_true');a=p.parse_args();history={}
    if a.interval<1:p.error('interval must be positive')
    run=a.run.resolve();last={}
    try:
        while True:
            now=time.time();state=read(WORK/'auto_eval'/run.name/'status.json');phase=state.get('phase','等待启动')
            gpus=state.get('physical_gpus',[0,1,2,3]);workers=state.get('workers',[]);manager=process(state.get('watcher_pid'))
            labels=dict(evaluating='四卡全量推理',verifying_step2000='校验最终学生权重',completed='全量推理已完成',
                        watcher_stopped_or_failed='推理已停止，请检查日志')
            overall=labels.get(phase,phase)
            if phase!='completed' and manager['alive'] is False:overall='推理管理进程已退出'
            lines=[f'A1 LIBERO | 推理 GPU {",".join(map(str,gpus))} | '+datetime.now(ZoneInfo('Asia/Singapore')).strftime('%m-%d %H:%M:%S UTC+8'),
                f'状态：{overall} | 管理进程 {state.get("watcher_pid","-")} {manager["label"]} | 每 {a.interval:g}s 刷新',
                '教师 Stage I  1000/1000 已完成 | 学生 Stage II 2000/2000 已完成',
                '全量推理 '+bar(state.get('episodes_completed',0),2000),
                f'任务 {state.get("tasks_completed",0)}/40 | 成功 {state.get("successes",0)}/{state.get("episodes_completed",0)}'+
                    (f' ({state["success_rate"]:.2%}，阶段性结果)' if state.get('success_rate') is not None else ''),
                '']
            usage=gpu_usage(gpus)
            for gpu in gpus:
                item=next((x for x in workers if x['gpu']==gpu),dict(gpu=gpu,phase='queued'))
                pid=item.get('server_pid');alive=process(pid);calls=item.get('calls',0);episodes=item.get('episodes_completed',0)
                key=(gpu,pid,item.get('suite'),item.get('task_id'))
                prior=history.get(key);delta=calls-prior[0] if prior else 0;delta_ep=episodes-prior[1] if prior else 0
                history[key]=(calls,episodes,now)
                fresh=now-state.get('updated_at',0)<30
                if phase=='completed':activity='已完成'
                elif alive['alive'] is False and pid:activity='进程已退出'
                elif phase=='verifying_step2000':activity='校验权重，等待加载'
                elif not fresh:activity='状态未更新，请检查'
                elif delta>0 or delta_ep>0:activity=f'持续推进：+{delta} 次动作请求，+{delta_ep} 回合'
                else:activity=LABELS.get(item.get('phase'),item.get('phase','等待启动'))
                lines.append(f'GPU {gpu} | 模型 PID {pid or "-"} {alive["label"]} | {activity}')
                if item.get('suite') is not None:
                    lines.append(f'  {item["suite"]} task {item["task_id"]} '+bar(episodes,50,16)+f' | 动作请求 {calls}')
                if gpu in usage:
                    u=usage[gpu];mem=u.get('processes',{}).get(pid)
                    lines.append(f'  模型显存 {mem:.1f} GiB | 全卡 {u["used"]:.1f}/{u["total"]:.0f} GiB | 利用率 {u["utilization"]}%' if mem is not None
                                 else f'  全卡显存 {u["used"]:.1f}/{u["total"]:.0f} GiB')
                if item.get('error'):lines.append('  '+item['error'])
            lines+=['',f'状态更新距今 {max(0,int(now-state.get("updated_at",now)))} 秒 | 保存已完成回合，任务可续跑',
                    'Ctrl+b 然后 d：离开，后台继续 | 窗口 2：并行推理日志']
            if state.get('error'):lines.append('错误：'+state['error'].strip().splitlines()[-1])
            if sys.stdout.isatty() and not a.once:
                import shutil
                size=shutil.get_terminal_size((140,36));lines=[clip(x,max(20,size.columns-1)) for x in lines]
                sys.stdout.write('\033[H\033[2J')
            print('\n'.join(lines),flush=True)
            if a.once:break
            time.sleep(a.interval)
    except KeyboardInterrupt:pass
if __name__=='__main__':main()
