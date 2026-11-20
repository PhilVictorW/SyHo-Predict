"""No GPU/model/simulator: test deadline boundaries, failure classes and real CPU GRPO continuation."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import sys,unittest,tempfile,json,subprocess,urllib.error
from pathlib import Path
from unittest.mock import patch,Mock
from types import SimpleNamespace
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[3]))
import timeout_runtime as rt
from stage_III.robotwin.common import atomic_json,read_json,WORK,load_checkpoint
from stage_III.robotwin.rpc import ModelError

class Timeouts(unittest.TestCase):
 def test_timeout_classification_is_specific(self):
  for e in (TimeoutError("x"),subprocess.TimeoutExpired("worker",5),urllib.error.URLError(TimeoutError("x")),ModelError('Model service rejected the request: '+json.dumps({"error":"TimeoutError('slow')"}))):self.assertTrue(rt.is_timeout(e))
  for e in (RuntimeError("timeout somewhere"),ModelError('Model service rejected the request: '+json.dumps({"error":"ValueError('wrong shape')"})),ValueError("bad action")) :self.assertFalse(rt.is_timeout(e))
  old=repr(ModelError('Model service rejected the request: '+json.dumps({"error":"TimeoutError('slow')"})))
  self.assertTrue(rt.failure_is_timeout({"kind":"model_error","error":old}))
 def test_no_progress_boundary_and_log_activity(self):
  for elapsed,expires in ((2399,False),(2401,True)):
   proc=Mock(pid=123,returncode=0);proc.poll.side_effect=[None,0]
   state={"pid":123,"phase":"executing","last_progress_at":100,"updated_at":100+elapsed}
   with patch.object(rt,'read_json',return_value=state),patch.object(rt.time,'time',return_value=100+elapsed),patch.object(rt,'stop_owned') as stop:
    if expires:
     with self.assertRaises(TimeoutError):rt.wait_group(proc,'unused',rt.POLICY,99999,100)
     stop.assert_called_once_with(proc)
    else:rt.wait_group(proc,'unused',rt.POLICY,99999,100);stop.assert_not_called()
 def test_actual_progress_extends_deadline(self):
  proc=Mock(pid=123,returncode=0);proc.poll.side_effect=[None,0]
  with patch.object(rt,'read_json',return_value={"pid":123,"phase":"executing","last_progress_at":2500}),patch.object(rt.time,'time',return_value=3000),patch.object(rt,'stop_owned') as stop:
   rt.wait_group(proc,'unused',rt.POLICY,99999,100);stop.assert_not_called()
 def test_setup_and_hard_budget_timeouts(self):
  for phase,last,hard in (("initializing",100,99999),("executing",1900,1000)):
   proc=Mock(pid=123,returncode=0);proc.poll.return_value=None
   with patch.object(rt,'read_json',return_value={"pid":123,"phase":phase,"last_progress_at":last}),patch.object(rt.time,'time',return_value=2000),patch.object(rt,'stop_owned'):
    with self.assertRaises(TimeoutError):rt.wait_group(proc,'unused',rt.POLICY,hard,100)
 def runtime(self,folder):
  x=rt.OnlineRuntime.__new__(rt.OnlineRuntime);x.out=Path(folder);x.url='http://127.0.0.1:1';x.pending=None;x.worker=None
  x.ensure_ready=Mock();x.recover_after_timeout=Mock();return x
 def arguments(self,folder):
  cfg={"runtime":{"sim_python":"fake","sim_root":folder,"max_infrastructure_retries":1,"hard_trajectory_seconds":100}}
  return cfg,'config',dict(task='task',mode='demo_clean'),dict(url='http://127.0.0.1:1',actor_sha256='a'),0,4,3,Path(folder)/'group'
 def test_timeout_no_retry_then_next_group_completes(self):
  with tempfile.TemporaryDirectory() as tmp:
   x=self.runtime(tmp);args=self.arguments(tmp)
   with patch.object(rt,'spawn',return_value=Mock()) as spawn,patch.object(rt,'child_env',return_value={}),patch.object(rt,'wait_group',side_effect=TimeoutError('stall')):
    result=x.collect(*args);self.assertEqual(result['status'],'skipped');self.assertEqual(result['trajectories_used'],0);self.assertEqual(spawn.call_count,1)
   args=list(args);args[4]=1;args[-1]=Path(tmp)/'next'
   def completed(*a):atomic_json(Path(tmp)/'next/result.json',dict(status='completed',trajectories=[{}]*4))
   with patch.object(rt,'spawn',return_value=Mock()),patch.object(rt,'child_env',return_value={}),patch.object(rt,'wait_group',side_effect=completed):self.assertEqual(x.collect(*args)['status'],'completed')
 def test_model_loading_timeout_is_nonfatal(self):
  with tempfile.TemporaryDirectory() as tmp:
   x=self.runtime(tmp);x.ensure_ready.side_effect=TimeoutError('load')
   with patch.object(rt,'spawn') as spawn:self.assertEqual(x.collect(*self.arguments(tmp))['status'],'skipped');spawn.assert_not_called()
 def test_child_reported_rpc_timeout_skips_without_retry(self):
  with tempfile.TemporaryDirectory() as tmp:
   x=self.runtime(tmp);args=self.arguments(tmp)
   atomic_json(Path(tmp)/'group/failure.json',dict(time=1e20,kind='model_error',error=repr(ModelError('Model service rejected the request: '+json.dumps({"error":"TimeoutError('slow')"})))))
   with patch.object(rt,'spawn',return_value=Mock()) as spawn,patch.object(rt,'child_env',return_value={}),patch.object(rt,'wait_group',side_effect=RuntimeError('child exit')):
    self.assertEqual(x.collect(*args)['status'],'skipped');self.assertEqual(spawn.call_count,1)
 def test_real_model_error_is_not_silenced(self):
  with tempfile.TemporaryDirectory() as tmp:
   x=self.runtime(tmp);atomic_json(Path(tmp)/'group/failure.json',dict(time=1e20,kind='model_error',error="ValueError('wrong shape')"))
   with patch.object(rt,'spawn',return_value=Mock()),patch.object(rt,'child_env',return_value={}),patch.object(rt,'wait_group',side_effect=RuntimeError('child exit')):
    with self.assertRaisesRegex(RuntimeError,'Non-timeout model error'):x.collect(*self.arguments(tmp))
 def test_adopted_rollout_is_not_restarted(self):
  with tempfile.TemporaryDirectory() as tmp:
   x=self.runtime(tmp);args=self.arguments(tmp);row,req=args[2:4]
   digest=rt.canonical_hash(dict(row=row,request={k:v for k,v in req.items() if k!='url'},group=0,n=4))
   x.pending=dict(output=str((Path(tmp)/'group').resolve()),identity_hash=digest,process={},started_at=0)
   def completed(*a):atomic_json(Path(tmp)/'group/result.json',dict(status='completed',trajectories=[{}]*4))
   with patch.object(rt,'spawn') as spawn,patch.object(rt,'AdoptedProcess',return_value=SimpleNamespace(pid=100)),patch.object(rt,'wait_group',side_effect=completed):
    self.assertEqual(x.collect(*args)['status'],'completed');spawn.assert_not_called()

class Continuation(unittest.TestCase):
 def test_whole_timed_out_iteration_saves_and_continues(self):
  import torch
  import train_direct_gpu3_v2 as driver
  from stage_III.robotwin.gate_actor import GateActor
  from stage_III.robotwin.tests.test_core import cache_group
  record=read_json(HERE.parent/'latest_gpu1.json');source=Path(record['run_dir'])
  cfg=driver.load_config(HERE/'workflow_direct_gpu3.json');cfg['rl'].update(iterations=2,eval_every=1)
  class FakeRuntime:
   def __init__(self,*args):self.url='fake';self.actor=None
   def set_actor(self,path,digest):
    with torch.random.fork_rng(devices=[]):self.actor=GateActor();self.actor.load_state_dict(load_checkpoint(path)['actor'])
   def collect(self,cfg,config,row,request,gid,n,gpu,out):
    if gid<4:return dict(status='skipped',reason='synthetic timeout')
    with torch.random.fork_rng(devices=[]):
     torch.manual_seed(gid);g=cache_group(self.actor)
    for t in g:t.update(actor_sha256=request['actor_sha256'],student_sha256=request['student_sha256'],group_id=gid)
    return dict(status='completed',trajectories=g)
   def close(self):pass
  def verify(a,cfg,out,manifest):
   atomic_json(out/'startup.json',{'test':True});atomic_json(out/'runtime_revision_v2.json',{'test':True})
  with tempfile.TemporaryDirectory(prefix='timeout_driver_cpu_',dir=WORK/'runs') as tmp:
   base=Path(tmp);student=base/'student';student.write_text('synthetic')
   a=SimpleNamespace(train_manifest=str(source/'manifests/prepared_train.json'),run_dir=str(base/'run'),student=str(student),resume=None,config=str(HERE/'workflow_direct_gpu3.json'),gpu='3',adopt_runtime=None)
   with patch.object(driver,'verify_startup',verify),patch.object(rt,'OnlineRuntime',FakeRuntime):driver.run(a,cfg)
   one=load_checkpoint(base/'run/checkpoints/actor_0001.pt');two=load_checkpoint(base/'run/checkpoints/actor_0002.pt')
   self.assertEqual(one['update_count'],0);self.assertEqual(one['total_episodes'],0)
   self.assertEqual(two['update_count'],1);self.assertEqual(two['total_episodes'],16)
   self.assertEqual(two['timeout_policy']['execution_no_progress_seconds'],2400)
   self.assertEqual(read_json(base/'run/status.json')['phase'],'completed_pending_final_comparison')
   self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]);result=unittest.TextTestRunner(verbosity=2).run(suite)
 atomic_json(HERE/'timeout_tests.json',dict(tests=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),scope='CPU and process mocks; no live model or GPU processes stopped'))
 if not result.wasSuccessful():raise SystemExit(1)
