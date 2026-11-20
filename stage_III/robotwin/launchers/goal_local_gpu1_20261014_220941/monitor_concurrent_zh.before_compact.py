"""只读：GPU3 Stage III与GPU7场景验证并行进度。"""
import datetime,time,subprocess,sys
from pathlib import Path
from monitor_zh import read,jsonl,bar,spark,number,alive
HERE=Path(__file__).resolve().parent

def render():
 record=read(HERE.parent/"latest_gpu1.json");run=Path(record["run_dir"]);rl=run/"rl"
 status=read(rl/"status.json");metrics=jsonl(rl/"metrics.jsonl");last=metrics[-1] if metrics else {}
 scene_status=read(run/"workflow_status.json");route=read(HERE/"gpu_route.json")
 labels={"loading":"加载模型", "collecting":"在线采样", "iteration_committed":"本轮已保存", "completed_pending_final_comparison":"采样训练完成，等待独立评估"}
 lines=["═"*90,"RoboTwin：GPU3 Stage III 奖励训练  ║  GPU7 场景验证（独立推进）",datetime.datetime.now().strftime("更新时间 %Y-%m-%d %H:%M:%S  |  每5秒刷新  |  Ctrl+b 然后 d 退出查看"),"", "【GPU3 强化学习】"]
 lines += ["阶段："+labels.get(status.get("phase"),status.get("phase","正在启动")),bar(last.get("iteration",0),100),f"计划：50任务×2模式，每模式4组×4条轨迹，共1600条；每轮保存Actor。",f"已提交有效轨迹：{last.get('total_episodes',0)}/1600  |  实际Actor更新：{last.get('update_count',0)}  |  环境步：{last.get('total_env_steps',0)}"]
 if status.get("task"):lines.append(f"当前第{status.get('iteration',0)+1}轮，第{status.get('group')}组：{status['task']} / {status.get('mode')}")
 skips=jsonl(rl/"skipped_groups.jsonl")
 lines.append(f"已跳过采样组：{len(skips)}（超时组不参与更新，后续训练继续）")
 if skips:lines.append("最近跳过："+str(skips[-1].get("task"))+" / "+str(skips[-1].get("mode"))+"："+str(skips[-1].get("reason"))[:160])
 service=read(rl/"model_service.json");pid=service.get("pid")
 if pid:lines.append(f"GPU3模型服务 PID {pid}："+("存活" if alive(pid) else "已退出"))
 for p in sorted((rl/"groups").glob("group_*/progress.json"))[-2:]:
  s=read(p);row=read(p.parent/"row.json")
  lines.append(f"  {p.parent.name} {row.get('task','')}：{s.get('phase')}，第{s.get('trajectory',0)+1}/4条，环境步{s.get('env_steps',0)}，距进展{max(0,int(time.time()-s.get('last_progress_at',time.time())))}秒")
 successes=sum(x.get("successes",0) for x in metrics);episodes=last.get("total_episodes",0)
 if episodes:lines.append(f"采样轨迹成功率 {successes}/{episodes} = {successes/episodes:.2%}（训练数据，不能作为测试精度）")
 for key,title in (("loss","Actor loss"),("reward_mean","平均奖励"),("kl","KL"),("eta_mean","eta均值")):
  vals=[x[key] for x in metrics if x.get(key) is not None];lines += [f"{title}：{number(vals[-1] if vals else None)}", "  "+spark(vals)]
 cp=read(rl/"last_checkpoint.json");lines += ["最近检查点："+str(cp.get("path","尚未保存")),"", "【GPU7 场景验证】",f"状态：{scene_status.get('phase')} / {scene_status.get('status')}"]
 for split,title,n in (("train","训练",200),("val","验证",300),("test","测试",300)):
  m=read(run/f"manifests/prepared_{split}.json");lines.append(title+"场景 "+bar(len(m.get("rows",[])),n)+("  已齐备" if m.get("complete") else ""))
 lines += ["", "训练使用已验收的200个训练场景；验证/测试场景不参与梯度更新。", "GPU3：加载后40分钟无进展跳过；采样超时不退出训练。GPU7检查规则保持。"]
 failure=read(rl/"failure.json")
 if failure:lines += ["【训练异常】"+str(failure)]
 try:
  text=subprocess.check_output(["nvidia-smi","-i","3,7","--query-gpu=index,memory.used,memory.free,utilization.gpu","--format=csv,noheader"],text=True,timeout=4)
  lines += ["", "GPU序号 / 已用显存 / 剩余显存 / 利用率",text.strip()]
 except (OSError,subprocess.SubprocessError):pass
 return "\n".join(lines)
if __name__=="__main__":
 if "--once" in sys.argv:print(render())
 else:
  while True:
   print("\033[2J\033[H"+render(),flush=True);time.sleep(5)
