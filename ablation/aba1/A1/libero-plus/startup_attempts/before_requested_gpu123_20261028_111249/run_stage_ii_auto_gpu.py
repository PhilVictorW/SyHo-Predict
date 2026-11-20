"""Persistent eight-GPU admission, staggered Stage II, elastic checkpoints, full eval."""
import argparse,collections,fcntl,json,math,os,signal,subprocess,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(WORK),str(WORK.parents[3])]
from a1_local.config_runtime import load_config,run_root,atomic_json,checkpoint_path
from a1_local.preflight import check_complete,check_teacher
from ablation.aba1.plus_common.environment import subprocess_environment
import after_training_full_eval as evaluation
LOGS=WORK/'logs'
BUDGET=8704  # 8.5 GiB per rank, including CUDA context
RESERVE=4096
HOST_RESERVE=16384
FIRST_HOST=38*1024
EXTRA_HOST=20*1024

def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}

def host_available():
    return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))/1024

def worker_states(output):
    result=[]
    for p in output.glob('status_rank*.json'):
        s=read(p);pid=s.get('pid',0)
        if pid and evaluation.live_process(pid,'a1_local.training'):
            try:
                cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
                if str(WORK/'config.json') in cmd and '--worker' in cmd:result.append(s)
            except OSError:pass
    return result

def sample(output):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used','--format=csv,noheader,nounits'],text=True,timeout=15)
    states=worker_states(output);pids={s['pid'] for s in states};own=collections.defaultdict(int)
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=15)
    for line in procs.splitlines():
        parts=[s.strip() for s in line.split(',')]
        if len(parts)==3 and parts[1].isdigit() and int(parts[1]) in pids and parts[2].isdigit():own[parts[0]]+=int(parts[2])
    gpus={}
    for line in rows.splitlines():
        i,uuid,free,used=[s.strip() for s in line.split(',')]
        if 0<=int(i)<8:gpus[int(i)]=dict(free_mib=int(free),used_mib=int(used),own_mib=own[uuid],capacity_mib=int(free)+own[uuid])
    if set(gpus)!=set(range(8)):raise RuntimeError('Expected all eight physical GPUs')
    rss=0
    for pid in pids:
        try:rss+=int(next(s.split()[1] for s in Path(f'/proc/{pid}/status').read_text().splitlines() if s.startswith('RssAnon:')))/1024
        except (OSError,StopIteration):pass
    return dict(time=time.time(),gpus=gpus,host_available_mib=host_available(),own_host_mib=rss,workers=states)

def choose_mapping(capacities,host_mib,current=(),blocked=()):
    limits={int(g):max(0,int((free-RESERVE)//BUDGET)) if int(g) not in blocked else 0 for g,free in capacities.items()}
    host_limit=0 if host_mib<HOST_RESERVE+FIRST_HOST else 1+int((host_mib-HOST_RESERVE-FIRST_HOST)//EXTRA_HOST)
    limit=min(32,host_limit,sum(limits.values()))
    counts=collections.Counter();mapping=[]
    for g in current:
        if len(mapping)<limit and counts[g]<limits.get(g,0):mapping.append(g);counts[g]+=1
    while len(mapping)<limit:
        choices=[g for g in limits if counts[g]<limits[g]]
        if not choices:break
        g=max(choices,key=lambda k:(capacities[k]-counts[k]*BUDGET,-k))
        mapping.append(g);counts[g]+=1
    return mapping,dict(gpu_slots=limits,host_limit=host_limit,rank_budget_mib=BUDGET,gpu_reserve_mib=RESERVE,
        host_reserve_mib=HOST_RESERVE,host_capacity_mib=host_mib,capacity_mib=capacities)

class Scheduler:
    def __init__(self,args):
        self.args=args;self.cfg=load_config(WORK/'config.json','A1');self.run=run_root(self.cfg);self.out=self.run/'II'
        self.history=collections.deque();self.blocked={};self.child=None;self.mapping=[];self.generation=0
        self.started=time.time();self.last_launch=0;self.pending=None;self.failures=0;self.log_path=None
        self.owns_locks=False;self.teacher=checkpoint_path(self.cfg,'I',3000)
    def record(self,phase,**kw):
        state=dict(time=time.time(),pid=os.getpid(),phase=phase,monitor_script=str(Path(__file__).resolve()),armed=True,
            poll_seconds=self.args.poll_seconds,physical_gpus=sorted(set(self.mapping)),rank_to_gpu=self.mapping,
            training_ranks=len(self.mapping),generation=self.generation,stage_I_steps=3000,stage_II_steps=2500,
            evaluation_workers=6,evaluation_tasks=10030,evaluation_gpus=self.cfg['a1_training']['physical_gpus'],
            checkpoint_policy='every 50 steps, retain latest complete only',**kw)
        atomic_json(LOGS/'gpu_scheduler_status.json',state);atomic_json(LOGS/'auto_continuation_status.json',state)
    def event(self,name,**kw):
        row=dict(time=time.time(),event=name);row.update(kw)
        with (LOGS/'gpu_scheduler_events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def capacities(self,s):
        self.history.append(s)
        while self.history and self.history[0]['time']<s['time']-600:self.history.popleft()
        cap={g:min(x['gpus'][g]['capacity_mib'] for x in self.history) for g in range(8)}
        # Incorporate the observed GPU0 transient from the earlier real-update profile.
        prior=read(LOGS/'gpu0_capacity_selection.json')
        if prior and s['time']-prior.get('time',0)<1800:cap[0]=min(cap[0],prior['peak_adjusted_free_mib'])
        return cap
    def latest(self):
        choices=sorted(self.out.glob('step_*/checkpoint_complete.json'),reverse=True)
        for marker in choices:
            try:
                meta,_=check_complete(marker.parent,require_resume=True)
                if meta['config']==self.cfg and meta['extra']['stage']=='II':return marker.parent,int(meta['step'])
            except (OSError,ValueError,KeyError):continue
        return None,0
    def archive_runtime(self):
        dest=WORK/'startup_attempts'/f'auto_gpu_{time.strftime("%Y%m%d_%H%M%S")}_g{self.generation}'
        dest.mkdir(parents=True,exist_ok=True)
        for p in self.out.glob('*.json'):
            if p.name not in ('latest.json','recovery_retention.json'):p.replace(dest/p.name)
        receipt=LOGS/'stage_II_exit_code.txt'
        if receipt.exists():receipt.replace(dest/receipt.name)
    def launch(self,mapping,detail):
        checkpoint,step=self.latest();self.generation+=1;self.archive_runtime()
        self.mapping=list(mapping);devices=list(dict.fromkeys(mapping));self.pending=None;self.last_launch=time.time()
        self.log_path=LOGS/f'train_stageII_auto_g{self.generation}_{time.strftime("%Y%m%d_%H%M%S")}.log'
        command=[self.cfg['model_python'],'-B','-u','-m','a1_local.training','--config',str(WORK/'config.json'),
            '--stage','II','--gpus',','.join(map(str,devices)),'--rank-to-gpu',','.join(map(str,mapping)),
            '--elastic-runtime','--workers','1','--cpu-threads','2','--offload-saved-tensors','--share-frozen-encoders',
            '--keep-models','1','--minimum-free-mib',str(BUDGET+RESERVE),'--execute']
        if checkpoint:command+=['--resume',str(checkpoint)]
        env=subprocess_environment(self.cfg,','.join(map(str,devices)),self.run/'runtime_cache')
        env.update(PYTHONPATH=str(WORK)+os.pathsep+env['PYTHONPATH'],PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            TORCH_GLOO_LAZY_INIT='1',MUJOCO_EGL_DEVICE_ID='0',TMPDIR='/tmp')
        with open(os.devnull) as stdin,self.log_path.open('a') as log:
            self.child=subprocess.Popen(command,cwd=WORK,env=env,stdin=stdin,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=tuple(f.fileno() for f in self.locks))
        (LOGS/'nohup_training.pid').write_text(str(self.child.pid)+'\n')
        atomic_json(LOGS/'launch_record.json',dict(time=time.time(),stage='II',physical_gpus=devices,world_size=len(mapping),
            rank_to_gpu=mapping,launcher_pid=self.child.pid,launcher_sid=self.child.pid,launch_method='nohup',
            resume_step=step,teacher=str(self.teacher),stage_I_steps=3000,stage_II_steps=2500,capacity=detail,
            model_loading='serialized_file_lock',save_every=50,keep_checkpoints_per_stage=1,training_log=str(self.log_path),
            continuation=dict(enabled=True,pid=os.getpid(),stages=['II:2500','LIBERO-Plus full:10030'])))
        atomic_json(self.run/'pipeline_status.json',dict(phase='training',stage='II',time=time.time(),rank_to_gpu=mapping,world_size=len(mapping)))
        self.event('launched',rank_to_gpu=mapping,resume_step=step,child_pid=self.child.pid,command=command,capacity=detail)
        subprocess.run(['tmux','respawn-window','-k','-t','a1_gpu012356_progress:1',f'exec tail -F {self.log_path}'],capture_output=True)
    def stop_owned(self):
        states=worker_states(self.out)
        if self.child and self.child.poll() is None:
            try:os.killpg(self.child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for s in states:
            try:os.kill(s['pid'],signal.SIGTERM)
            except ProcessLookupError:pass
        if self.child:
            try:self.child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid,signal.SIGKILL);self.child.wait(timeout=10)
        for s in worker_states(self.out):
            try:os.kill(s['pid'],signal.SIGKILL)
            except ProcessLookupError:pass
    def full_eval(self):
        check_complete(checkpoint_path(self.cfg,'II',2500),verify_hash=True)
        receipt=LOGS/'stage_II_exit_code.txt';receipt.write_text('0\n')
        atomic_json(self.run/'pipeline_status.json',dict(phase='completed',completed_stages=['I','II'],teacher_step=3000,student_step=2500,time=time.time()))
        command=[self.cfg['model_python'],'-B','-u',str(WORK/'after_training_full_eval.py'),'--execute','--workers','6',
            '--poll-seconds','10','--training-exit-file',str(receipt)]
        with open(os.devnull) as stdin:
            self.child=subprocess.Popen(command,cwd=WORK,stdin=stdin,start_new_session=True,pass_fds=(self.locks[0].fileno(),))
        while self.child.poll() is None:
            result=read(evaluation.STATE);self.record('full_evaluation',completed=result.get('completed',0),child_pid=self.child.pid)
            time.sleep(self.args.poll_seconds)
        result=read(evaluation.STATE)
        if self.child.returncode or result.get('phase')!='completed' or result.get('completed')!=10030:
            raise RuntimeError('Full evaluation did not complete all 10030 tasks')
        self.record('completed',completed=10030,success_rate=result.get('success_rate'))
    def run_forever(self):
        self.locks=[(LOGS/'.auto_continuation.lock').open('a'),(self.run/'.pipeline.lock').open('a')]
        for lock in self.locks:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.owns_locks=True
        if worker_states(self.out):raise RuntimeError('Existing training workers must be inspected before adoption')
        (LOGS/'gpu_scheduler.pid').write_text(str(os.getpid())+'\n');(LOGS/'auto_continuation.pid').write_text(str(os.getpid())+'\n')
        check_teacher(self.teacher,self.cfg);check_complete(self.teacher,verify_hash=True)
        atomic_json(self.run/'pipeline_status.json',dict(phase='waiting_capacity',stage='II',time=time.time()))
        self.event('monitor_started',budget_mib=BUDGET,reserve_mib=RESERVE,gpus=list(range(8)),bootstrap_seconds=60)
        critical_count=0
        while True:
            try:s=sample(self.out)
            except (OSError,subprocess.SubprocessError,ValueError) as e:
                self.event('sample_error',error=repr(e));time.sleep(self.args.poll_seconds);continue
            cap=self.capacities(s);blocked=[g for g,t in self.blocked.items() if t>s['time']]
            host=min(x['host_available_mib']+x['own_host_mib'] for x in self.history if x['time']>=s['time']-60)
            desired,detail=choose_mapping(cap,host,self.mapping,blocked)
            with (LOGS/'gpu_scheduler_samples.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
            if self.child and self.child.poll() is not None:
                code=self.child.returncode;checkpoint,step=self.latest()
                if code==0 and step==2500:self.full_eval();return
                ready=read(self.out/'elastic_ready.json')
                if code==0 and ready.get('step')==step and step>0:
                    self.event('checkpoint_reconfiguration',step=step);self.child=None;self.mapping=[]
                    continue
                log=self.log_path.read_text(errors='replace')[-30000:] if self.log_path else ''
                memory_failure=any(x in log.lower() for x in ('out of memory','memory admission floor','cannot allocate memory','memoryerror'))
                if not memory_failure:raise RuntimeError(f'Training exited {code}; inspect {self.log_path}')
                self.stop_owned();self.event('memory_retry',exit_code=code,resume_step=step)
                for g in self.mapping:self.blocked[g]=s['time']+300
                self.child=None;self.mapping=[];self.failures+=1
                time.sleep(15);continue
            if self.child is None:
                _,step=self.latest()
                if step==2500:self.full_eval();return
                if desired and s['time']-self.history[0]['time']>=60:
                    self.launch(desired,detail)
                else:self.record('waiting_capacity',resources=s,capacity=detail,candidate_rank_to_gpu=desired)
            else:
                states=s['workers'];step=max([x.get('step',0) for x in states]+[0])
                loaded=len(states)==len(self.mapping) and all(x.get('phase') in ('training','checkpoint','reconfigure_ready') for x in states)
                critical=any(s['gpus'][g]['free_mib']<1024 for g in set(self.mapping)) or s['host_available_mib']<4096
                critical_count=critical_count+1 if critical else 0
                if critical_count>=2:
                    self.event('pressure_stop',step=step,resources=s)
                    self.stop_owned()
                    for g in self.mapping:self.blocked[g]=s['time']+300
                    self.child=None;self.mapping=[];critical_count=0;continue
                # Resizing is collective: preserve the current checkpoint and rejoin all ranks.
                if loaded and desired!=self.mapping and s['time']-self.last_launch>180:
                    request=dict(time=s['time'],minimum_step=((step//50)+1)*50,rank_to_gpu=desired,
                        reason='stable capacity change',generation=self.generation)
                    if not self.pending:
                        atomic_json(self.out/'elastic_reconfigure.json',request);self.pending=request
                        self.event('resize_requested',**request)
                self.record('stage_II',stage_II_step=step,child_pid=self.child.pid,training_pids=[x['pid'] for x in states],
                    resources=s,capacity=detail,candidate_rank_to_gpu=desired,pending_reconfiguration=self.pending,
                    training_log=str(self.log_path))
            time.sleep(self.args.poll_seconds)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--poll-seconds',type=float,default=5)
    args=p.parse_args()
    if args.poll_seconds<2:p.error('Poll interval must be at least two seconds')
    scheduler=Scheduler(args)
    if not args.execute:
        s=sample(scheduler.out);cap={g:v['capacity_mib'] for g,v in s['gpus'].items()}
        mapping,detail=choose_mapping(cap,s['host_available_mib'])
        print(json.dumps(dict(snapshot=s,candidate=mapping,capacity=detail,training_started=False),indent=2));return
    try:scheduler.run_forever()
    except BaseException as error:
        if scheduler.owns_locks:scheduler.record('failed',error=repr(error))
        raise
if __name__=='__main__':main()
