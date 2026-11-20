"""Read-only 4-suite x 7-category tmux evaluation dashboard."""
from __future__ import annotations
import argparse
import datetime
import os
from pathlib import Path
import time
from .eval_common import read, SUITES, CATEGORIES, GPUS


def bar(done,total,width=30):
    n=round(width*min(1,done/total)) if total else 0
    return '['+'#'*n+'-'*(width-n)+']'


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--interval',type=float,default=5);p.add_argument('--once',action='store_true')
    args=p.parse_args();folder=args.folder.resolve()
    names=dict(prepared='衔接准备完成',waiting_training='等待Stage III训练完成',verifying='校验最终权重',
        starting_servers='加载八路评估模型',evaluating='全量推理进行中',completed='全量推理已完成',
        completed_with_errors='遍历结束，仍有超时/错误项',failed='异常，请查看评估日志')
    labels=['Spatial','Object','Goal','Long-10'];heads=['Camera','Robot','Language','Light','Background','Noise','Layout']
    while True:
        status=read(folder/'status.json');protocol=read(folder/'protocol.json');result=read(folder/'results.json')
        phase=status.get('phase','prepared');train=read(Path(protocol['train_run'])/'status.json') if protocol else {}
        lines=['Stage III 第一轮最终模型 | LIBERO-plus 8380项推理 | 4 suite × 7类别',
               f'时间 {datetime.datetime.now():%m-%d %H:%M:%S} | {names.get(phase,phase)} | 每{args.interval:g}秒刷新']
        pid=status.get('pid');alive=bool(pid and Path(f'/proc/{pid}').exists())
        if phase not in ('completed','completed_with_errors','prepared') and not alive:
            lines.append('警告：衔接/评估主进程不在运行，请检查日志；面板刷新不代表评估继续。')
        lines.append(f'训练 {train.get("batch",0)}/262批，已提交 {train.get("groups_processed",0)}/8380组 | '
                     '评估模型：Student4000 + 最终GateActor262（Beta均值）')
        n=result.get('processed',0);total=8380;done=result.get('done',0);success=result.get('successes',0)
        lines.append(f'总进度 {bar(n,total)} {n}/{total} ({100*n/total:.2f}%) | 正常完成 {done} | 成功 {success}')
        lines.append(f'超时 {result.get("timeout",0)} | 错误 {result.get("error",0)} | 运行 {result.get("running",0)} | '
                     f'待运行 {result.get("pending",8380)} | 未完成项最多3次尝试')
        lines.append('各格进度：已处理/计划（处理含超时、错误）；等待训练期间均为0')
        lines.append(f'{"Suite":8}|'+'|'.join(f'{h:^12}' for h in heads))
        cells=result.get('cells',{})
        for suite,label in zip(SUITES,labels):
            values=[]
            for c in CATEGORIES:
                cell=cells.get(suite,{}).get(c,{})
                processed=cell.get('done',0)+cell.get('timeout',0)+cell.get('error',0)
                expected=cell.get('expected',0)
                values.append(f'{processed}/{expected} {100*processed/expected if expected else 0:.0f}%')
            lines.append(f'{label:8}|'+'|'.join(f'{v:^12}' for v in values))
        lines.append('各格成绩：成功/正常完成；超时和错误不计入正常完成分母')
        lines.append(f'{"Suite":8}|'+'|'.join(f'{h:^12}' for h in heads))
        for suite,label in zip(SUITES,labels):
            values=[]
            for c in CATEGORIES:
                cell=cells.get(suite,{}).get(c,{})
                a,b=cell.get('successes',0),cell.get('done',0)
                rate=f'{100*a/b:.1f}%' if b else '--'
                values.append(f'{a}/{b} {rate}')
            lines.append(f'{label:8}|'+'|'.join(f'{v:^12}' for v in values))
        lines.append('-'*100)
        for gpu in (1,3,6,7):
            slots=[i for i,g in enumerate(GPUS) if g==gpu];texts=[];card_done=0
            for slot in slots:
                w=result.get('workers',{}).get(str(slot),{});card_done+=w.get('done',0)
                heartbeat=read(folder/'workers'/f'worker_{slot}.json');active=w.get('active')
                if active:
                    age=max(0,int(time.time()-heartbeat.get('updated_at',time.time())))
                    text=f'S{slot} g{active["id"]} {heartbeat.get("phase","启动")} {age}s'
                elif phase=='waiting_training':text=f'S{slot} 等训练结束'
                elif phase=='starting_servers':
                    ready=read(folder/'servers'/f'server_{slot}_ready.json')
                    text=f'S{slot} '+('已就绪' if ready.get('gate_mode')=='beta_mean' else '加载中')
                else:text=f'S{slot} 等待/已完成'
                texts.append(text)
            lines.append(f'GPU{gpu} 完成{card_done}项 | '+' ; '.join(texts))
        rate=result.get('success_rate')
        lines.append(f'已完成项成功率：{100*rate:.2f}%' if rate is not None else '已完成项成功率：--')
        if result.get('complete'):lines.append(f'完整8380成功率：{100*success/8380:.2f}%')
        else:lines.append('完整8380成功率：待全部正常完成；当前为训练任务上的复测结果。')
        if status.get('error'):lines.append('异常：'+str(status['error'])[:100])
        lines.append('查看日志：tmux attach -t stage3_plus_round1_eval | 返回后台：Ctrl-b 再按 d')
        print('\033[2J\033[H'+'\n'.join(lines),flush=True)
        if args.once:return
        time.sleep(max(1,args.interval))


if __name__=='__main__':main()
