#!/usr/bin/env python3
"""本次实验的只读中文监控，不导入模型、不管理训练进程。"""
import argparse
import collections
import datetime
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import time

PHASES = [
    ("offline_check","配置与路径检查"),("candidate_manifest","生成候选场景清单"),
    ("interaction","Goal→Local 交互训练"),("sensitivity","交互敏感性检查"),
    ("prepare_train","准备训练场景"),("prepare_val","准备验证场景"),
    ("prepare_test","准备测试场景"),("paired_validation","配对在线验证"),
    ("admission","Stage III 准入判定"),("stage_III","Stage III 奖励训练"),
    ("final_comparison","最终对照评测")]
LABEL = dict(PHASES)
SUB = {"loading_base":"正在校验并构建组合模型","loading":"正在加载冻结模型服务",
       "interaction":"正在训练交互模块","collecting":"正在采集同初态轨迹",
       "iteration_committed":"本轮更新与权重已保存",
       "interaction_complete_pending_online_acceptance":"交互训练完成，等待在线验收",
       "stopped_by_offline_regression":"离线误差触发退化阈值，已停止交互训练",
       "stopped_at_interaction_budget":"达到本轮交互预算，已停止",
       "completed_pending_final_comparison":"Stage III 完成，等待最终对照"}

def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {} if default is None else default

def jsonl(path):
    try:lines=Path(path).read_text().splitlines()
    except OSError:return []
    result=[]
    for line in lines:
        try:result.append(json.loads(line))
        except ValueError:pass
    return result

def number(v):
    try:return f"{float(v):.6f}" if math.isfinite(float(v)) else str(v)
    except (TypeError,ValueError):return "尚无数据"

def bar(n,total):
    f=min(1,max(0,n/max(1,total)));k=int(f*32)
    return "["+"█"*k+"░"*(32-k)+f"] {n}/{total} ({100*f:.1f}%)"

def spark(values):
    values=[float(x) for x in values if isinstance(x,(int,float)) and math.isfinite(x)]
    if not values:return "尚无训练记录"
    if len(values)>64:
        values=[sum(values[i*len(values)//64:(i+1)*len(values)//64])/len(values[i*len(values)//64:(i+1)*len(values)//64]) for i in range(64)]
    lo,hi=min(values),max(values);chars="▁▂▃▄▅▆▇█"
    return "".join(chars[min(7,int(7*(x-lo)/max(hi-lo,1e-12)))] for x in values)

def alive(pid):
    try:
        text=(Path("/proc")/str(pid)/"stat").read_text()
        return text.split(") ",1)[1].split()[0]!="Z"
    except (OSError,IndexError):return False

def duration(seconds):
    sec=max(0,int(seconds));return f"{sec//3600}小时{sec//60%60:02d}分{sec%60:02d}秒"

class Display:
    def __init__(self,record):
        self.record=record;self.run=Path(record["run_dir"]);self.launch=Path(record["launcher"])
        self.cfg=read(record["config"]);self.seen=read(self.launch/"中文监控记录.json",{"seen":[]})["seen"]
        self.seen=set(self.seen);self.gpu="等待读取 GPU 状态";self.gpu_at=0;self.own_mib={}
        self.events=self.launch/"中文事件日志.log"
        self.train_pid=None
    def event(self,key,message):
        if key in self.seen:return
        with self.events.open("a") as f:
            f.write("["+datetime.datetime.now().strftime("%m-%d %H:%M:%S")+"] "+message+"\n")
        self.seen.add(key)
        p=self.launch/"中文监控记录.json";tmp=p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"seen":sorted(self.seen)},ensure_ascii=False));tmp.replace(p)
    def refresh_gpu(self):
        if time.time()-self.gpu_at<15:return
        self.gpu_at=time.time()
        try:
            s=subprocess.check_output(["nvidia-smi","-i",str(self.record.get("gpu",1)),"--query-gpu=memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True,timeout=4).strip()
            used,free,util=[float(x.strip()) for x in s.split(",")]
            self.gpu=f"GPU{self.record.get('gpu',1)} 总占用 {used/1024:.1f} GiB | 剩余 {free/1024:.1f} GiB | 利用率 {util:.0f}%"
            s=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],text=True,timeout=4)
            self.own_mib={int(x.split(",")[0]):float(x.split(",")[1]) for x in s.splitlines()}
        except (OSError,ValueError,subprocess.SubprocessError):self.gpu="暂时无法读取 GPU 状态，训练不受监控读取影响"

    def render(self):
        run=self.run;w=read(run/"workflow_status.json");ist=read(run/"interaction/status.json")
        rst=read(run/"rl/status.json");im=jsonl(run/"interaction/metrics.jsonl");rm=jsonl(run/"rl/metrics.jsonl")
        phase=w.get("phase","启动中");state=w.get("status","等待状态")
        active=read(self.launch/"active_execution.json")
        if "gpu" in active:
            self.record["gpu"]=active["gpu"]
            self.event("resume:"+str(active.get("started_at")),f"用户已指定 GPU{active['gpu']}，后台续跑已启动；复用交互200步权重，补齐场景并验收后进入Stage III。")
        route=read(self.launch/"gpu_route.json")
        if route:
            self.event("gpu_route:"+str(route.get("updated_at")),f"GPU安排已更新：检查继续在GPU{route['checks_gpu']}，验收通过后在GPU{route['training_gpu']}进行Stage III训练及后续评测。当前检查进程保持运行。")
        fix=read(self.launch/"reset_bounds_fix.json")
        if fix.get("real_scene_validation")=="passed":
            self.event("reset_bounds_fix_passed","已修复环境重置时桌面偏移累积问题；同种子物理状态和三相机图像完全一致。新环境重新验场景，旧记录保留，交互200步权重不变。")
        self.train_pid=ist.get("pid");self.refresh_gpu()
        scene_lines=[]
        for split,title,quota in [("train","训练",route.get("task_mode_cells",16)*2),("val","验证",route.get("task_mode_cells",16)*3),("test","测试",route.get("task_mode_cells",16)*3)]:
            m=read(run/f"manifests/prepared_{split}.json")
            if m or phase.startswith("prepare_"):
                scene_lines.append(f"{title}场景：{len(m.get('rows',[]))}/{quota}；尝试 {len(m.get('events',[]))}；"+("已齐备" if m.get("complete") else "待补齐"))
            for ev in m.get("events",[]):
                result={"accepted":"通过","rejected":"不可用，改用后续候选","skipped":"超时或异常，已跳过"}.get(ev["status"],ev["status"])
                self.event("scene:"+ev["scene_id"]+":"+ev["status"],title+"场景 "+ev["scene_id"]+"："+result)
        if state=="paused_by_user":
            self.event("user_pause:gpu_selection","已停止本次自动工作流和仿真；保留交互200步权重，等待用户明确指定GPU。不会自动恢复。")
        ready=read(self.launch/"readiness.json")
        if ready.get("status")=="offline_ready_pending_gpu_checks":
            self.event("offline_resume_ready","离线准备已核对；指定GPU后接着完成场景与在线验收，通过后开始Stage III。")
        self.event("start","后台工作流已启动：GPU1；先训练交互200步，再验收，通过后进入Stage III。")
        mapping=read(self.launch/"simulator_device_check.json")
        if mapping.get("matches"):self.event("sim_gpu_verified","仿真渲染器与CUDA均指向GPU1，设备映射检查通过。")
        self.event("phase:"+phase+":"+state,"当前工作阶段："+LABEL.get(phase,phase)+"；状态："+{"running":"进行中","stopped":"已停止","comparison_completed":"已完成","paused_by_user":"已暂停，等待用户指定GPU"}.get(state,state))
        checks=[
          ("配置检查",run/"static_check.json"),
          ("候选场景清单",run/"manifests/candidates.json"),
          ("组合权重身份校验",run/"interaction/base_snapshot.json"),
          ("数据采样与 episode 划分",run/"interaction/sampling.json"),
          ("真实前向/梯度/教师筛查报告",run/"interaction/preflight.json"),
          ("第0步离线动作误差评估",run/"interaction/validation_000000.json")]
        for title,path in checks:
            if path.exists():self.event("file:"+str(path),title+"报告已生成。")
        if im:
            last=im[-1];step=last["step"]
            self.event("step:"+str(step),f"交互训练已完成第{step}步；action={number(last.get('action'))}，Local蒸馏={number(last.get('kd_local'))}，Goal蒸馏={number(last.get('kd_goal'))}。")
        saved=[]
        for p in sorted((run/"interaction/checkpoints").glob("student_*.pt.complete.json")):
            step=int(p.name.split("_")[1].split(".")[0]);saved.append(step)
            self.event("ckpt:"+str(step),f"交互训练第{step}步权重已完整保存（包含校验标记）。")
        for row in rm:
            i=row["iteration"];self.event("rl:"+str(i),f"Stage III 第{i}轮已提交；是否更新Actor={row.get('updated')}，KL={number(row.get('kl'))}，累计有效episodes={row.get('total_episodes',0)}。")
        failure=[]
        for sub in ("interaction","rl"):
            f=read(run/sub/"failure.json")
            if f:
                message=sub+"异常："+str(f.get("error",f))
                self.event("failure:"+message,message);failure.append(message)
        lines=[
            "═"*78,
            "RoboTwin 中文进度台：Goal→Local 交互训练 → 验收 → Stage III",
            datetime.datetime.now().strftime("更新时间：%Y-%m-%d %H:%M:%S")+"（每5秒刷新；Ctrl+b 后按 d 可退出查看）",
            f"本次实验：{self.record['name']}",
            f"已运行：{duration(time.time()-self.record.get('launched_at',time.time()))}",
            self.gpu,
            "当前阶段："+LABEL.get(phase,phase)+"；"+{"running":"进行中","stopped":"已停止","comparison_completed":"已完成","paused_by_user":"已暂停，等待用户指定GPU"}.get(state,state)]
        substate=ist if phase=="interaction" else rst if phase=="stage_III" else {}
        if substate:
            detail=SUB.get(substate.get("phase"),substate.get("phase",""))
            if phase=="interaction" and not im:
                if (run/"interaction/validation_000000.json").exists():detail="基线评估已完成，正在准备第0步检查点或首个优化步"
                elif (run/"interaction/preflight.json").exists():detail="前向/梯度/教师筛查报告已生成，正在评估基线动作误差"
                elif (run/"interaction/sampling.json").exists():detail="采样索引已生成，正在解码验证样本或进行前向/梯度检查"
                elif (run/"interaction/base_snapshot.json").exists():detail="权重身份已核对，模型构建/组合权重加载或数据索引处理中"
            lines.append("正在做什么："+detail)
            pid=substate.get("pid")
            if pid:
                lines.append(f"进程：PID {pid}，"+("存活" if alive(pid) else "未运行")+"；该进程GPU显存："+(f"{self.own_mib[pid]/1024:.2f} GiB" if pid in self.own_mib else "尚未出现在GPU计算进程列表"))
            if pid and alive(pid):
                try:
                    proc=Path("/proc")/str(pid)
                    state_code=(proc/"stat").read_text().split(") ",1)[1].split()[0]
                    io=dict(line.split(": ",1) for line in (proc/"io").read_text().splitlines())
                    wait="正在等待磁盘I/O" if state_code=="D" else "正常运行/等待下一次调度"
                    if phase=="interaction" and not im:
                        fields=dict(line.split(":",1) for line in (proc/"status").read_text().splitlines() if ":" in line)
                        rss=int(fields.get("VmRSS","0 kB").split()[0])/2**20
                        self.event("init_minute:"+str(int(time.time()//60)),
                            f"模型初始化仍在进行：PID {pid}，内存 {rss:.2f} GiB，累计磁盘读取 {int(io['read_bytes'])/2**30:.2f} GiB；{wait}，尚未完成第1步。")
                    lines.append(f"资源状态：{wait}；累计物理读取 {int(io['read_bytes'])/2**30:.2f} GiB")
                except (OSError,ValueError,KeyError):pass
            if phase=="interaction" and not im:lines.append("优化器更新：尚未完成第1步。后台启动不等于已经开始训练更新。")
        lines += ["","【整体流程】"]
        keys=[x[0] for x in PHASES];idx=keys.index(phase) if phase in keys else len(keys) if phase in ("pilot_completed","full_completed") else -1
        stages=[]
        for i,(key,title) in enumerate(PHASES):
            mark="已完成" if i<idx else ("已暂停" if state=="paused_by_user" else "异常停止" if state=="stopped" else "进行中") if i==idx else "待开始"
            stages.append(f"{i+1:02d}.[{mark}] {title}")
        width=shutil.get_terminal_size((160,48)).columns
        group=3 if width>=150 else 2 if width>=100 else 1
        lines.extend("  "+"  |  ".join(stages[i:i+group]) for i in range(0,len(stages),group))
        lines += ["","【交互训练】",bar(im[-1]["step"] if im else 0,self.cfg["interaction"]["steps"]),
                  "已完整保存步数："+(", ".join(map(str,saved)) if saved else "暂无"),
                  "损失曲线从第1步开始；数值越低的点显示越矮，曲线长度较大时按区间均值压缩。"]
        for key,title in [("action","action loss"),("kd_local","Local 蒸馏"),("kd_goal","Goal 蒸馏"),("keep","原模型保持损失")]:
            vals=[row[key] for row in im if key in row]
            lines.append(f"  {title}: 当前 {number(vals[-1] if vals else None)} | 首步 {number(vals[0] if vals else None)}")
            lines.append("    "+spark(vals))
        if im:lines.append("最新门控 scale："+", ".join(f"{x:.5f}" for x in im[-1].get("gate_scales",[])))
        pre=read(run/"interaction/preflight.json")
        if pre:lines.append("教师/基线 action loss 比值："+number(pre.get("teacher_to_baseline_action_loss_ratio"))+"；该报告不是在线成功率。")
        lines+=["","【Stage III 与在线验收】",bar(rm[-1]["iteration"] if rm else 0,self.cfg["rl"]["iterations"])]
        if not rm:lines.append("尚无奖励训练更新；必须先通过交互敏感性和配对在线验收。")
        else:
            row=rm[-1];lines.append(f"累计有效episodes {row.get('total_episodes',0)}；环境步 {row.get('total_env_steps',0)}；KL {number(row.get('kl'))}；eta均值 {number(row.get('eta_mean'))}")
        admission=read(run/"admission.json")
        if admission:lines.append("准入结果："+("通过" if admission.get("passed") else "未通过")+"；配对差异 "+number(admission.get("statistics",{}).get("mean_difference")))
        for split,title in [("train","训练"),("val","验证"),("test","测试")]:
            m=read(run/f"manifests/prepared_{split}.json")
            if m:lines.append(f"{title}场景：有效 {len(m.get('rows',[]))}，已尝试 {len(m.get('events',[]))}，配额"+("完整" if m.get("complete") else "未完成"))
        evaldirs=[run/"paired_validation",*sorted((run/"rl").glob("validation_*")),run/"final_comparison/val_controls",run/"final_comparison/test"]
        for folder in evaldirs[-2:] if phase=="final_comparison" else evaldirs:
            e=read(folder/"report.json");pr=read(folder/"progress.json")
            if e:
                lines.append("评测："+str(folder.relative_to(run))+"；"+("覆盖完整" if e.get("complete") else "覆盖不完整"))
                for arm,rows in e.get("arms",{}).items():
                    n=len(rows);win=sum(bool(r["success"]) for r in rows)
                    lines.append(f"  {arm}: {win}/{n}（{100*win/n:.2f}%）" if n else f"  {arm}: 尚无完成episode")
            elif pr:lines.append(f"在线评测 {folder.name}：{pr.get('arm')} 已处理 {pr.get('completed')}/{pr.get('total')} 场景；成功 {pr.get('successes')}；跳过 {pr.get('skipped')}（未完成前不作最终精度）")
        if failure:lines+=["","【需要处理的异常】",*failure]
        lines += ["","【最近中文事件】"]
        try:lines += self.events.read_text().splitlines()[-10:]
        except OSError:pass
        lines+=["","历史91.45%是旧基线已完成组合的成绩；本轮新成功率只以在线结果为准。",
                "完整中文历史：切换至窗口1「中文事件」查看；所有说明保留在启动目录。","═"*78]
        try:height=os.get_terminal_size(1).lines-1
        except OSError:height=47
        height=max(18,height)
        if len(lines)>height:
            current=im[-1] if im else {}
            compact=[f"RoboTwin 中文进度 | GPU{self.record.get('gpu',1)} | "+datetime.datetime.now().strftime("%m-%d %H:%M:%S"),
                "当前："+LABEL.get(phase,phase)+"；"+{"running":"进行中","stopped":"异常停止","paused_by_user":"已暂停，等待用户指定GPU"}.get(state,state),
                "正在做："+("等待指定GPU；不会自动恢复" if state=="paused_by_user" else detail if substate else LABEL.get(phase,phase)),
                self.gpu,
                "交互训练 "+bar(current.get("step",0),self.cfg["interaction"]["steps"])]
            for key,title in [("action","动作损失"),("kd_local","Local蒸馏"),("kd_goal","Goal蒸馏"),("keep","保持损失")]:
                vals=[r[key] for r in im if key in r]
                compact.append(title+"："+number(current.get(key))+"  "+spark(vals))
            compact.extend(["已保存步数："+(",".join(map(str,saved)) if saved else "暂无"),
                "Stage III "+bar(rm[-1]["iteration"] if rm else 0,self.cfg["rl"]["iterations"])])
            if route:compact.append(f"GPU安排：检查GPU{route['checks_gpu']} → 奖励训练/后续评测GPU{route['training_gpu']}")
            compact.extend(scene_lines)
            done=[title for title,path in checks if path.exists()]
            compact.append("已完成："+" / ".join(done[:3]))
            if len(done)>3:compact.append("已完成："+" / ".join(done[3:]))
            if substate.get("pid"):compact.append("本阶段PID："+str(substate["pid"])+"；"+("存活" if alive(substate["pid"]) else "未运行"))
            if phase=="interaction" and not im:
                compact.append("尚未完成第1个优化步；loss会在真实更新后显示。")
                if substate.get("pid") in self.own_mib:compact.append("本进程已分配GPU显存："+f"{self.own_mib[substate['pid']]/1024:.2f} GiB")
            if failure:compact.append("异常："+failure[-1])
            compact.append("最近完成/当前动作（完整历史在窗口1）：")
            try:events=self.events.read_text().splitlines()
            except OSError:events=[]
            room=max(0,height-len(compact)-1)
            compact.extend(events[-room:] if room else [])
            compact.append("Ctrl+b n 切换中文事件；Ctrl+b d 退出查看，后台继续。")
            lines=compact
        return "\n".join(lines)

def main():
    p=argparse.ArgumentParser();p.add_argument("--record",required=True);p.add_argument("--once",action="store_true");p.add_argument("--events-only",action="store_true")
    a=p.parse_args();d=Display(read(a.record));offset=0
    while True:
        try:
            if a.events_only:
                if d.events.exists():
                    with d.events.open() as f:f.seek(offset);text=f.read();offset=f.tell()
                    print(text,end="",flush=True)
            else:print(("" if a.once else "\033[2J\033[H")+d.render(),flush=True)
        except Exception as exc:print("中文监控读取暂时失败："+repr(exc)+"；监控不会中断训练。",flush=True)
        if a.once:return
        time.sleep(5)
if __name__=="__main__":main()
