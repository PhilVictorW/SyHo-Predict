"""Read-only 4-suite x 7-category LIBERO-Plus dashboard."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

WORK=Path(__file__).resolve().parent
RUN=WORK/'runs/single_latent16_dino_step1000_2gpu07_20261019'
FOLDER=RUN/'evaluation/full_epoch_0000_step_000001000'
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']
CATEGORIES=['Camera','Robot','Language','Light','Background','Noise','Layout']

def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}

def bar(done,total,width=4):
    k=min(width,int(width*done/max(1,total)))
    return '#' * k + '-' * (width-k)

class Dashboard:
    def __init__(self):
        cfg=read(WORK/'config.json')
        self.tasks=read(Path(cfg['full_eval_manifest']))['tasks']
        self.expected=Counter((x['suite'],x['category']) for x in self.tasks)
        self.wanted={(x['suite'],x['task_id']):x for x in self.tasks}
        self.rows={}
        self.mtimes={}

    def render(self):
        workflow=read(WORK/'workflow_full_eval.json');training=read(RUN/'status.json')
        protocol=read(FOLDER/'protocol.json')
        episode_dir=FOLDER/'episodes'
        if episode_dir.exists():
            for entry in os.scandir(episode_dir):
                if not entry.name.endswith('.json'):continue
                stamp=entry.stat().st_mtime_ns
                if self.mtimes.get(entry.name)==stamp:continue
                row=read(Path(entry.path));key=(row.get('suite'),row.get('task_id'))
                if (key in self.wanted and not row.get('error') and
                    row.get('protocol_sha256')==protocol.get('protocol_sha256') and
                    row.get('category')==self.wanted[key]['category']):
                    self.rows[key]=row;self.mtimes[entry.name]=stamp
        done=Counter((r['suite'],r['category']) for r in self.rows.values())
        success=sum(bool(r['success']) for r in self.rows.values())
        count=len(self.rows);total=len(self.tasks)
        states={}
        if (FOLDER/'queue.sqlite').exists():
            try:
                with sqlite3.connect((FOLDER/'queue.sqlite').as_uri()+'?mode=ro',uri=True,timeout=1) as db:
                    states=dict(db.execute('SELECT state,COUNT(*) FROM tasks GROUP BY state'))
            except sqlite3.Error:pass
        labels={'waiting_for_step1000':'等待训练及1000步保存',
                'waiting_for_training_exit':'等待训练进程退出并释放显存',
                'verifying_step1000':'校验1000步权重与恢复分片',
                'deleting_step750':'已验证1000步，正在删除750步',
                'starting_adaptive_full_evaluation':'全量推理 / 模型加载',
                'completed':'全量推理完成','failed':'流程失败'}
        phase=workflow.get('phase','waiting_for_step1000')
        screen=[f'A0 LIBERO-Plus 全量推理 | {time.strftime("%Y-%m-%d %H:%M:%S")}',
            '状态：'+labels.get(phase,phase),
            f'训练 {training.get("step","--")}/1000 | 推理 [{bar(count,total,30)}] {count}/{total} ({count/total:.1%})']
        if phase=='verifying_step1000':
            n=workflow.get('verified_bytes',0);size=workflow.get('file_bytes',1)
            screen.append(f'校验 {workflow.get("file","")}：{n/2**30:.1f}/{size/2**30:.1f} GiB')
        placed=read(FOLDER/'placement.json')
        if placed:
            workers=Counter(map(int,placed['worker_gpus']))
            screen.append('选中 GPU：'+', '.join(f'{g}（最多{n}路）' for g,n in workers.items()))
        else:screen.append('选卡：训练退出后，按剩余显存选择最多的四张 GPU；每卡最多3路')
        final=read(RUN/'final.json')
        old=RUN/'checkpoints/epoch_0000_step_000000750'
        screen.append('1000步：'+('已保存' if final.get('step')==1000 else '等待保存')+
            ' | 750步：'+('保留中' if old.exists() else '已删除')+
            f' | 磁盘剩余 {shutil.disk_usage(WORK).free/2**30:.1f} GiB')
        width=max(8,min(14,(shutil.get_terminal_size((130,30)).columns-13)//7-1))
        def line(label,values):
            return f'{label:<11}|'+'|'.join(f'{str(x):^{width}}' for x in values)+'|'
        headers=['Camera','Robot','Language','Light','Backgrnd','Noise','Layout']
        screen += [line('Suite',headers),'-'*min(125,12+7*(width+1))]
        for suite in SUITES:
            label=suite.removeprefix('libero_') if suite!='libero_10' else 'libero_10'
            screen.append(line(label,[f'{done[suite,c]}/{self.expected[suite,c]}' for c in CATEGORIES]))
            screen.append(line('',[bar(done[suite,c],self.expected[suite,c],3)+f' {100*done[suite,c]/self.expected[suite,c]:.0f}%' for c in CATEGORIES]))
        running=[]
        for path in FOLDER.glob('scheduler_worker*.json'):
            row=read(path);running.append(row.get('phase','unknown'))
        counts=Counter(running)
        screen += [f'队列：完成 {states.get("done",0)} | 运行 {states.get("running",0)} | 待跑 {states.get("pending",total)} | 错误 {states.get("failed",0)}',
            f'评测进程：运行 {counts["running"]} | 加载 {counts["loading_model"]} | 等待 {counts["waiting_for_memory_and_load_slot"]}']
        if count:
            label='最终成功率' if count==total else '已完成样本成功率（非最终）'
            screen.append(f'{label}：{success}/{count} = {success/count:.2%}')
        if workflow.get('error'):screen.append('错误：'+workflow['error'])
        screen.append('每5秒刷新；Ctrl+b 然后 d 离开，不停止后台推理。Ctrl+b n 切换日志。')
        return '\n'.join(screen)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');args=p.parse_args()
    dashboard=Dashboard()
    while True:
        try:screen=dashboard.render()
        except Exception as e:screen='面板读取中：'+repr(e)
        print(('' if args.once else '\033[H\033[2J')+screen,flush=True)
        if args.once:break
        time.sleep(5)
