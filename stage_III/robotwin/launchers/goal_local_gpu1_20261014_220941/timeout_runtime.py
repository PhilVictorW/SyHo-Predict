"""Owned Stage III runtime: sampling timeouts skip groups, never escape to training."""
from __future__ import annotations
import json
import ast
import os
import subprocess
import time
import urllib.error
from pathlib import Path
from stage_III.robotwin.common import atomic_json, canonical_hash, read_json, sha256
from stage_III.robotwin.engine import start_server
from stage_III.robotwin.runtime import spawn, child_env, stop_owned
from stage_III.robotwin.rpc import call, ModelError

HERE = Path(__file__).resolve().parent
POLICY = dict(version=2, setup_no_progress_seconds=1800, execution_no_progress_seconds=2400,
              rpc_timeout_seconds=2400, timeout_action="skip_group_without_retry",
              partial_trajectories_used=False, checks_gpu7_policy="unchanged")

def process_identity(pid):
    try:
        p=Path('/proc')/str(pid); f=(p/'stat').read_text().split(') ',1)[1].split()
        return dict(pid=int(pid),start=f[19],state=f[0],uid=p.stat().st_uid,pgid=int(f[2]),
                    command=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode())
    except (OSError, ValueError, IndexError): return None

class AdoptedProcess:
    """An orphaned owned child; output artifacts, not an unavailable exit code, prove success."""
    def __init__(self, record):
        self.record=record; self.pid=record['pid']; self.returncode=None
        now=process_identity(self.pid)
        if now and now['state']!='Z':
            if any(now[k]!=record[k] for k in ('pid','start','uid','pgid','command')):
                raise ValueError('Adopted process identity changed')
            if now['uid']!=os.getuid() or now['pgid']!=now['pid']:
                raise ValueError('Cannot adopt an unrelated process group')
    def poll(self):
        now=process_identity(self.pid)
        if now and now['start']==self.record['start'] and now['state']!='Z':return None
        self.returncode=0; return 0
    def wait(self,timeout=None):
        begin=time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic()-begin>=timeout:
                raise subprocess.TimeoutExpired(self.record['command'],timeout)
            time.sleep(.1)
        return self.returncode

def is_timeout(exc):
    if isinstance(exc,(TimeoutError,subprocess.TimeoutExpired)):return True
    if isinstance(exc,urllib.error.URLError):return isinstance(exc.reason,TimeoutError)
    if isinstance(exc,ModelError):
        prefix='Model service rejected the request: '
        if str(exc).startswith(prefix):
            try:return failure_is_timeout(json.loads(str(exc)[len(prefix):]))
            except (ValueError,TypeError):pass
    return False

def failure_is_timeout(failure):
    if failure.get('kind')=='sampling_timeout':return True
    # Match serialized exception types, never an arbitrary occurrence of "timeout".
    error=str(failure.get('error',''))
    if error.startswith('ModelError(') and error.endswith(')'):
        try:
            message=ast.literal_eval(error[len('ModelError('):-1])
            if isinstance(message,str):return is_timeout(ModelError(message))
        except (ValueError,SyntaxError):pass
    return error.startswith(('TimeoutError(', 'TimeoutExpired(', 'socket.timeout(')) or (
        error.startswith('URLError(') and ('TimeoutError(' in error or 'socket.timeout(' in error))

def wait_group(proc, progress, policy, hard_seconds, started_at=None):
    start=time.time() if started_at is None else started_at
    try:
        while proc.poll() is None:
            try:row=read_json(progress)
            except (OSError,ValueError):row={}
            if row.get('pid')!=proc.pid:row={}
            last=max(start,float(row.get('last_progress_at',start)))
            setup=row.get('phase','initializing')=='initializing'
            limit=policy['setup_no_progress_seconds' if setup else 'execution_no_progress_seconds']
            if time.time()-last>limit:
                stage='sampling_setup' if setup else 'post_load_task'
                raise TimeoutError(f'{stage}: no meaningful task progress for {limit} seconds')
            if time.time()-start>hard_seconds:raise TimeoutError('Sampling hard runtime budget exceeded')
            try:proc.wait(timeout=2)
            except subprocess.TimeoutExpired:pass
        if proc.returncode:raise RuntimeError(f'Rollout exited with code {proc.returncode}')
    except BaseException:
        stop_owned(proc)
        raise

class OnlineRuntime:
    def __init__(self,cfg,config,student,gpu,out,adoption=None):
        self.cfg=cfg;self.config=config;self.student=student;self.gpu=gpu;self.out=Path(out)
        self.proc=None;self.worker=None;self.actor_path=None;self.actor_hash=None
        self.url=f"http://127.0.0.1:{cfg['runtime']['port']}"
        self.adoption=adoption;self.pending=adoption.get('rollout') if adoption else None
        self.student_hash=sha256(student);self.known_actor=None
        if adoption:
            self.proc=AdoptedProcess(adoption['server'])
            if adoption['student_sha256']!=self.student_hash:raise ValueError('Adopted Student mismatch')
            self.known_actor=adoption['actor_sha256']
            if self.proc.poll() is not None and self.pending:
                raise RuntimeError('Cannot adopt live rollout without its model service')
        atomic_json(self.out/'timeout_policy.json',POLICY)

    def set_actor(self,path,digest):
        if sha256(path)!=digest:raise ValueError('Actor checkpoint changed')
        self.actor_path=Path(path);self.actor_hash=digest

    def ensure_ready(self):
        if self.proc is None or self.proc.poll() is not None:
            self.proc,self.url,info=start_server(self.cfg,self.config,self.student,self.gpu,self.out)
            if info['student_sha256']!=self.student_hash:raise ValueError('Wrong model service')
            self.known_actor=None
            print(f'[模型服务] GPU{self.gpu}权重已加载，PID {self.proc.pid}。',flush=True)
            atomic_json(self.out/'model_service.json',dict(**info,gpu=int(self.gpu),url=self.url))
        if self.known_actor!=self.actor_hash:
            ack=call(self.url,'/actor',dict(path=str(self.actor_path)),timeout=POLICY['rpc_timeout_seconds'])
            if ack['actor_sha256']!=self.actor_hash:raise ValueError('Actor reload identity mismatch')
            self.known_actor=self.actor_hash

    def recover_after_timeout(self):
        if self.proc is None:return
        try:
            health=call(self.url,'/health',{},timeout=5)
            healthy=health['student_sha256']==self.student_hash and health['actor_sha256']==self.actor_hash
        except (OSError,RuntimeError,ValueError):healthy=False
        if not healthy:
            stop_owned(self.proc);self.proc=None;self.known_actor=None
            print('[超时恢复] 仅回收本次GPU3无响应的模型服务，下一组重新加载。',flush=True)

    def skip(self,out,row,group,reason,kind):
        value=dict(status='skipped',reason=reason,skip_kind=kind,group=group,trajectories_used=0,
                   task=row['task'],mode=row['mode'],time=time.time(),timeout_policy=POLICY)
        atomic_json(Path(out)/'result.json',value)
        with (self.out/'skipped_groups.jsonl').open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
        print(f"[跳过并继续] 第{group+1}组 {row['task']} / {row['mode']}：{reason}",flush=True)
        return value

    def collect(self,cfg,config,row,request,group,n,gpu,out):
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        identity=dict(row=row,request={k:v for k,v in request.items() if k!='url'},group=group,n=n)
        digest=canonical_hash(identity);marker=out/'collection.json';result=out/'result.json'
        if marker.exists() and read_json(marker)['identity_hash']!=digest:
            raise ValueError('Cached collection belongs to a different behavior/scene')
        atomic_json(marker,dict(identity_hash=digest,identity=identity))
        if result.exists():
            value=read_json(result)
            if value.get('status')=='completed' and len(value['trajectories'])==n:return value
            if value.get('status')=='skipped':return value
            raise ValueError('Invalid cached result')
        try:self.ensure_ready()
        except BaseException as exc:
            if not is_timeout(exc):raise
            self.recover_after_timeout()
            return self.skip(out,row,group,repr(exc),'sampling_timeout')
        request=dict(request,url=self.url)
        atomic_json(out/'row.json',row);atomic_json(out/'request.json',request)
        cmd=[cfg['runtime']['sim_python'],'-B',str(HERE/'rollout_timeout_v2.py'),
             '--config',str(config),'--row',str(out/'row.json'),'--request',str(out/'request.json'),
             '--group',str(group),'--trajectories',str(n),'--output',str(out),'--gpu',str(gpu),'--execute']
        reason='no complete group'
        for attempt in range(cfg['runtime']['max_infrastructure_retries']+1):
            start=time.time()
            if self.pending and self.pending['output']==str(out.resolve()):
                pending=self.pending;self.pending=None
                if pending['identity_hash']!=digest:raise ValueError('In-flight rollout identity changed')
                proc=AdoptedProcess(pending['process']);start=pending['started_at']
                print(f'[接续] 保持当前GPU3采样进程 PID {proc.pid} 和已完成轨迹。',flush=True)
            else:
                proc=spawn(cmd,out/f'rollout_v2_{attempt:02d}.log',cfg['runtime']['sim_root'],child_env(cfg,gpu,out))
            self.worker=proc
            try:
                wait_group(proc,out/'progress.json',POLICY,cfg['runtime']['hard_trajectory_seconds']*n,start)
                if not result.exists():raise RuntimeError('Worker exited without complete result')
                value=read_json(result)
                if value.get('status')!='completed' or len(value.get('trajectories',[]))!=n:
                    raise RuntimeError('Missing complete group')
                return value
            except BaseException as exc:
                failure=read_json(out/'failure.json') if (out/'failure.json').exists() else {}
                if failure.get('time',0)<start:failure={}
                if is_timeout(exc) or (isinstance(exc,RuntimeError) and failure_is_timeout(failure)):
                    self.recover_after_timeout()
                    return self.skip(out,row,group,repr(exc)+' '+str(failure),'sampling_timeout')
                if not isinstance(exc,RuntimeError):raise
                if failure.get('kind') in ('model_error','invalid_policy'):
                    raise RuntimeError('Non-timeout model error: '+str(failure)) from exc
                reason=repr(exc)+' '+str(failure)
            finally:self.worker=None
        return self.skip(out,row,group,reason,'infrastructure_failure')

    def close(self):
        if self.worker is not None:stop_owned(self.worker)
        if self.proc is not None:stop_owned(self.proc)
