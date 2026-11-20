"""One fixed Stage III checkpoint per GPU, 40 tasks x 50 episodes, resumable."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, signal, socket, subprocess, threading, time, traceback
from pathlib import Path
from goal_local.eval_common import ROOT, BASELINE, PYTHON, SIM_PYTHON, SUITES, atomic_json, read_json
from .client import rpc
from .eval_common import summarize, task_path, task_result

class Runner:
    def __init__(self, manifest, item):
        self.manifest=manifest; self.item=item; self.output=Path(item["output"])
        self.output.mkdir(parents=True,exist_ok=True)
        self.stop=threading.Event(); self.mutex=threading.Lock()
        self.state=dict(phase="starting",round_index=item["round_index"],gpu=item["gpu"])
        self.server=None; self.child=None; self.server_log=None
        self.url=f"http://127.0.0.1:{item['port']}"
        self.env=os.environ.copy()
        self.env.update(CUDA_VISIBLE_DEVICES=str(item["gpu"]),MUJOCO_EGL_DEVICE_ID=str(item["gpu"]),
            MUJOCO_GL="egl",PYTHONUNBUFFERED="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
            TOKENIZERS_PARALLELISM="false",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
            LIBERO_HOME=str(BASELINE/"runtime/LIBERO"),LIBERO_CONFIG_PATH=str(BASELINE/"runtime/libero_config"),
            PYTHONPATH=os.pathsep.join([str(ROOT),str(BASELINE),str(BASELINE/"runtime/LIBERO")]))

    def emit(self,event,**fields):
        print(json.dumps(dict(event=event,time=time.time(),round_index=self.item["round_index"],gpu=self.item["gpu"],**fields)),flush=True)

    def refresh(self):
        with self.mutex:
            summary=summarize(self.output,self.item["round_index"],self.item["gpu"])
            atomic_json(self.output/"queue_status.json",dict(**self.state,pid=os.getpid(),time=time.time(),
                server_pid=self.server.pid if self.server else None,task_pid=self.child.pid if self.child else None,
                episodes_completed=summary["episodes_completed"],tasks_completed=summary["tasks_completed"]))
            return summary

    def heartbeat(self):
        while not self.stop.wait(10):
            try:self.refresh()
            except Exception:traceback.print_exc()

    @staticmethod
    def end(process):
        if process is not None and process.poll() is None:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:return
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()

    def load(self):
        self.end(self.server)
        if self.server_log:self.server_log.close()
        self.server=None
        with socket.socket() as probe:probe.bind(("127.0.0.1",self.item["port"]))
        free=int(subprocess.check_output(["nvidia-smi",f"--id={self.item['gpu']}","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
        if free<16000:raise RuntimeError(f"GPU {self.item['gpu']} has only {free} MiB free")
        ready=self.output/"server_ready.json";ready.unlink(missing_ok=True)
        self.state.update(phase="starting_server")
        self.server_log=(self.output/"server.log").open("a")
        self.server=subprocess.Popen([PYTHON,"-m","stage_III.libero.eval_server","--port",str(self.item["port"]),
            "--device","cuda:0","--ready",str(ready),"--checkpoint",self.manifest["base_student"],
            "--actor-checkpoint",self.item["actor_checkpoint"]],cwd=ROOT,env=self.env,
            stdout=self.server_log,stderr=subprocess.STDOUT,start_new_session=True)
        self.emit("server_started",pid=self.server.pid,port=self.item["port"])
        deadline=time.monotonic()+1800
        while time.monotonic()<deadline:
            if self.stop.is_set():raise RuntimeError("Stopping")
            if self.server.poll() is not None:raise RuntimeError(f"Server exited {self.server.returncode}; see server.log")
            if read_json(ready,{}).get("pid")==self.server.pid:
                rpc(self.url,"/health",timeout=10);break
            self.stop.wait(2)
        else:raise TimeoutError("Server startup timed out")
        self.state.update(phase="loading_student_and_actor");self.refresh()
        info=rpc(self.url,"/load",dict(checkpoint=self.manifest["base_student"],actor_checkpoint=self.item["actor_checkpoint"]),timeout=7200)
        assert info["checkpoint"]==self.manifest["base_student"] and info["actor_checkpoint"]==self.item["actor_checkpoint"]
        assert info["step"]==4000 and info["gate_mode"]=="beta_mean" and info["actor_precision"]=="float32"
        assert not any(info[k] for k in ("teacher_loaded","uses_state","uses_future"))
        self.emit("model_loaded",info=info)

    def run(self):
        lock=(self.output/"evaluation.lock").open("a")
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if self.manifest.get("base_student_verified_sha256")!=self.manifest["base_student_expected_sha256"]:
            raise RuntimeError("Base checkpoint verification is incomplete")
        assert hashlib.sha256(Path(self.item["actor_checkpoint"]).read_bytes()).hexdigest()==self.item["actor_sha256"]
        core=ROOT/"examples/LIBERO/eval_files/libero_eval_core.py"
        assert hashlib.sha256(core.read_bytes()).hexdigest()==self.manifest["rollout_core_sha256"]
        protocol=dict(**self.item,base_student=self.manifest["base_student"],
            base_student_sha256=self.manifest["base_student_verified_sha256"],rollout_core_sha256=self.manifest["rollout_core_sha256"],
            suites=list(SUITES),task_ids=list(range(10)),episodes_per_task=50,expected_episodes=2000,
            initial_state_indices=list(range(50)),simulator_seed=0,num_steps_wait=10,image_size=256,
            action_chunk=8,flow_steps=10,max_steps=dict(zip(SUITES,[250,300,320,550])),
            gate_mode="beta_mean",gate_formula="eta = 2 * alpha / (alpha + beta)",actor_precision="float32",
            student_precision="unchanged BF16 autocast",residual_precision="unchanged Stage3Policy: eta cast to delta dtype",
            action_noise_seed="(suite_index+1)*10000000+task_id*10000+episode_idx*100+chunk_index",
            gripper_conversion="historical core: 2*g-1",teacher_loaded=False,uses_state=False,uses_future=False,
            video_policy="At most one first successful episode per task; no failed episode videos",
            score="successes / 2000 after all 40 tasks complete",baseline_98_5_episode_count=400,
            note="Standard 0..49 initial states include Stage III training states 10..29; not a held-out test",
            started_at=time.time())
        atomic_json(self.output/"protocol.json",protocol)
        def shutdown(*_):
            self.state.update(phase="stopped");self.stop.set()
            self.end(self.child);self.end(self.server)
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
        self.refresh()
        thread=threading.Thread(target=self.heartbeat,daemon=True);thread.start()
        failures=[]
        try:
            if self.refresh()["status"]=="complete":
                self.state.update(phase="completed");return
            for attempt in range(1,4):
                try:self.load();break
                except Exception:
                    self.end(self.server)
                    if attempt==3:raise
                    traceback.print_exc();self.state.update(phase="retrying_server",attempt=attempt)
                    self.stop.wait(20)
            for suite in SUITES:
                for task in range(10):
                    if task_result(self.output,suite,task)["complete"]:continue
                    out=task_path(self.output,suite,task);out.mkdir(parents=True,exist_ok=True)
                    for attempt in range(1,4):
                        self.state.update(phase="evaluating",suite=suite,task_id=task,attempt=attempt)
                        self.refresh()
                        if self.server is None or self.server.poll() is not None:self.load()
                        with (out/"run.log").open("a") as log:
                            log.write(f"\n--- attempt {attempt} at {time.ctime()} ---\n");log.flush()
                            cmd=[SIM_PYTHON,"-m","stage_III.libero.eval_task","--checkpoint",self.manifest["base_student"],
                                "--actor-checkpoint",self.item["actor_checkpoint"],"--url",self.url,"--suite",suite,
                                "--task-id",str(task),"--episodes","50","--output",str(out)]
                            self.child=subprocess.Popen(cmd,cwd=ROOT,env=self.env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                            self.emit("task_started",suite=suite,task_id=task,attempt=attempt,pid=self.child.pid)
                            try:code=self.child.wait(timeout=21600)
                            except subprocess.TimeoutExpired:self.end(self.child);code=-9
                            self.child=None
                        result=task_result(self.output,suite,task)
                        if code==0 and result["complete"]:
                            self.emit("task_complete",**result);break
                        self.emit("task_retry",suite=suite,task_id=task,attempt=attempt,exit_code=code)
                        if attempt==3:failures.append(dict(suite=suite,task_id=task,exit_code=code))
                        else:self.stop.wait(5)
            summary=self.refresh()
            self.state.update(phase="completed" if summary["status"]=="complete" else "incomplete",failures=failures)
            assert hashlib.sha256(Path(self.item["actor_checkpoint"]).read_bytes()).hexdigest()==self.item["actor_sha256"]
            self.emit("evaluation_end",status=self.state["phase"],episodes=summary["episodes_completed"],successes=summary["successes"])
        except KeyboardInterrupt:
            self.state.update(phase="stopped");raise
        except Exception as exc:
            self.state.update(phase="failed",error=repr(exc));traceback.print_exc();raise
        finally:
            self.stop.set();self.end(self.child);self.end(self.server)
            self.child=self.server=None
            if self.server_log:self.server_log.close()
            thread.join(timeout=15);self.refresh();lock.close()
        if self.state["phase"]!="completed":raise RuntimeError("Some tasks remain incomplete; restart this command to resume")

def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--round",type=int,required=True)
    a=p.parse_args();manifest=json.loads(Path(a.manifest).read_text())
    item=next(x for x in manifest["rounds"] if x["round_index"]==a.round)
    Runner(manifest,item).run()
if __name__=="__main__":main()
