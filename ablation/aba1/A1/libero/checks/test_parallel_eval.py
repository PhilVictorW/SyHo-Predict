"""CPU checks for disjoint assignment, deterministic resume and serialized loading."""
import json
import queue
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W))
import parallel_evaluate as pe
class FakeProcess:
    counter=900000
    def __init__(self,code=None):
        FakeProcess.counter+=1;self.pid=FakeProcess.counter;self.returncode=code
    def poll(self):return self.returncode
    def wait(self,timeout=None):self.returncode=0;return 0
class Parallel(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=W/'checks');self.out=Path(self.temp.name);self.ckpt=self.out/'step_002000';self.ckpt.mkdir()
    def tearDown(self):self.temp.cleanup()
    def put_rows(self,suite,task,count):
        folder=pe.task_path(self.out,suite,task);folder.mkdir(parents=True,exist_ok=True)
        rows=[dict(task_id=task,episode_idx=i,success=True,num_actions=12) for i in range(count)]
        (folder/'episodes.checkpoint.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    def test_four_workers_resume_and_never_duplicate_tasks(self):
        self.put_rows(pe.SUITES[0],0,50);self.put_rows(pe.SUITES[0],1,17)
        original=(pe.task_path(self.out,pe.SUITES[0],1)/'episodes.checkpoint.jsonl').read_bytes()
        jobs=queue.Queue()
        for job in pe.pending_tasks(self.out):jobs.put(job)
        self.assertEqual(jobs.qsize(),39)
        stop=threading.Event();pool=pe.ServerPool([0,1,2,3],self.ckpt,self.out,'sha',stop,0)
        pool.ensure=lambda gpu:'http://fake'
        for entry in pool.entries.values():entry['server']=FakeProcess()
        seen=[];mutex=threading.Lock()
        def spawn(command,**kwargs):
            suite=command[command.index('--suite')+1];task=int(command[command.index('--task-id')+1])
            gpu=int(kwargs['env']['CUDA_VISIBLE_DEVICES'])
            self.assertEqual(kwargs['env']['MUJOCO_EGL_DEVICE_ID'],str(gpu))
            self.assertEqual(command[command.index('--episodes')+1],'50')
            with mutex:
                self.assertNotIn((suite,task),seen);seen.append((suite,task))
            out=Path(command[command.index('--output')+1]);existing=pe.episode_rows(out,task,50)
            with (out/'episodes.checkpoint.jsonl').open('a') as f:
                for i in range(50):
                    if i not in existing:f.write(json.dumps(dict(task_id=task,episode_idx=i,success=True,num_actions=12))+'\n')
            time.sleep(.005)
            return FakeProcess(0)
        workers=[pe.Worker(g,pool,stop) for g in [0,1,2,3]]
        with patch.object(pe.subprocess,'Popen',side_effect=spawn):
            with ThreadPoolExecutor(max_workers=4) as executor:
                results=list(executor.map(lambda w:w.run(jobs),workers))
        self.assertEqual(results,[[],[],[],[]]);self.assertEqual(len(seen),39)
        self.assertEqual(pe.pending_tasks(self.out),[])
        self.assertTrue((pe.task_path(self.out,pe.SUITES[0],1)/'episodes.checkpoint.jsonl').read_bytes().startswith(original))
        self.assertTrue(all(w.state['tasks_finished']>0 for w in workers))
        summary=pe.summarize(self.out,50,'sha');self.assertEqual(summary['episodes_completed'],2000)
    def test_models_load_one_at_a_time_on_distinct_gpus(self):
        stop=threading.Event();pool=pe.ServerPool([0,1,2,3],self.ckpt,self.out,'sha',stop,.025)
        mapping={};loading=[];times=[];mutex=threading.Lock()
        def spawn(command,**kwargs):
            p=FakeProcess();gpu=int(kwargs['env']['CUDA_VISIBLE_DEVICES'])
            ready=Path(command[command.index('--ready')+1]);ready.write_text(json.dumps(dict(pid=p.pid,port=p.pid)))
            mapping[p.pid]=gpu;return p
        def rpc(url,endpoint,payload):
            gpu=mapping[int(url.rsplit(':',1)[1])]
            with mutex:loading.append(gpu);self.assertEqual(len(loading),1);start=time.time()
            time.sleep(.025)
            with mutex:loading.remove(gpu);times.append((start,time.time()))
            return dict(variant='A1',step=2000,model_sha256='sha',physical_gpu=str(gpu),local_slots=16,goal_slots=0,
                        interaction=False,teacher_loaded=False,uses_future=False,uses_state=False)
        with patch.object(pe.subprocess,'Popen',side_effect=spawn),patch.object(pe,'rpc',side_effect=rpc):
            with ThreadPoolExecutor(max_workers=4) as executor:list(executor.map(pool.ensure,[0,1,2,3]))
        self.assertEqual(len(times),4)
        for first,second in zip(times,times[1:]):self.assertGreaterEqual(second[0]-first[1],.024)
    def test_checkpoint_or_protocol_change_is_rejected(self):
        meta=dict(config={'ablation':'A1'},extra={})
        with patch.object(pe,'model_metadata',return_value=meta),patch.object(pe,'file_sha256',return_value='rollout'):
            with self.assertRaisesRegex(ValueError,'protocol'):pe.validate_protocol(self.out,self.ckpt,'sha')
    def test_live_progress_ignores_partial_append_but_exact_resume_rejects(self):
        self.put_rows(pe.SUITES[0],0,3);p=pe.task_path(self.out,pe.SUITES[0],0)
        with (p/'episodes.checkpoint.jsonl').open('a') as f:f.write('{"unfinished":')
        self.assertEqual(len(pe.watch_rows(p)),3)
        with self.assertRaises(ValueError):pe.episode_rows(p,0,50)
if __name__=='__main__':unittest.main(verbosity=2)
