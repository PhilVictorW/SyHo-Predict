import json,tempfile,unittest,signal
from pathlib import Path
from unittest.mock import patch,Mock
import expand_gpu123_after_checkpoint as handoff

class Handoff(unittest.TestCase):
 def test_checkpoint50_before_six_rank_restart_and_only_old_monitor_signalled(self):
  with tempfile.TemporaryDirectory() as tmp:
   w=Path(tmp);logs=w/'logs';out=w/'run/II';logs.mkdir();out.mkdir(parents=True)
   flow=dict(pid=100,rank_to_gpu=[5,4,5]);(logs/'gpu_scheduler_status.json').write_text(json.dumps(flow))
   (logs/'launch_record.json').write_text(json.dumps(dict(launcher_pid=200)))
   (out/'contract.json').write_text(json.dumps(dict(code_sha256='known')))
   alive=True;calls=[];iteration=0;stages=[]
   def matches(pid,name):return alive if pid==100 else False
   def kill(pid,sig):
    nonlocal alive
    calls.append((pid,sig))
    if sig==signal.SIGKILL:alive=False
   def workers(path):
    nonlocal iteration
    iteration+=1
    return [dict(rank=0,pid=300,step=3 if iteration==1 else 48 if iteration==2 else 50)]
   def sample(path):
    if iteration>=3:
     (out/'elastic_ready.json').write_text(json.dumps(dict(step=50)))
     (out/'latest.json').write_text(json.dumps(dict(step=50,checkpoint=str(out/'step_000050'))))
    return dict(workers=[] if iteration>=3 else [dict(pid=300,step=48)],gpus={g:dict(free_mib=15000) for g in range(8)},host_available_mib=60000)
   fake=Mock();fake.run_forever.side_effect=lambda:stages.append('six ranks')
   with patch.object(handoff.base,'WORK',w),patch.object(handoff.base,'LOGS',logs),patch.object(handoff,'STATE',logs/'handoff.json'),patch.object(handoff.base,'load_config',return_value={}),patch.object(handoff.base,'run_root',return_value=w/'run'),patch.object(handoff.base,'worker_states',side_effect=workers),patch.object(handoff.base,'sample',side_effect=sample),patch.object(handoff,'matches',side_effect=matches),patch.object(handoff.os,'kill',side_effect=kill),patch.object(handoff,'code_identity',return_value={}),patch.object(handoff,'digest_json',return_value='known'),patch.object(handoff,'check_complete') as verify,patch.object(handoff,'RequestedScheduler',return_value=fake),patch.object(handoff.time,'sleep'),patch.object(handoff.sys,'argv',['expand_gpu123_after_checkpoint.py','--execute','--capacity-report',str(w/'report.json')]):
    handoff.main()
   self.assertEqual(calls,[(100,signal.SIGSTOP),(100,signal.SIGKILL)])
   self.assertEqual(stages,['six ranks'])
   verify.assert_called_once_with(out/'step_000050',require_resume=True,verify_hash=True)
   request=json.loads((out/'elastic_reconfigure.json').read_text())
   self.assertEqual(request['minimum_step'],50)
   self.assertEqual(request['rank_to_gpu'],[5,4,5,1,2,3])
if __name__=='__main__':unittest.main()
