"""CPU tests for six same-GPU model servers, task resume and the final-step gate."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from . import parallel_evaluation as pe

class FakeProcess:
    counter=900000
    def __init__(self,code=None):
        FakeProcess.counter+=1;self.pid=FakeProcess.counter;self.returncode=code
    def poll(self):return self.returncode

class ParallelEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=pe.WORK/'checks')
        self.root=Path(self.temp.name);self.out=self.root/'evaluation'/'result';self.out.mkdir(parents=True)
        self.run=self.root/'runs'/'test_run';self.ckpt=self.run/'stage_II/checkpoints/step_002000';self.ckpt.mkdir(parents=True)
        self.stop_patch=patch.object(pe,'stop_child',side_effect=lambda p: setattr(p,'returncode',0) if p else None)
        self.stop_patch.start()
    def tearDown(self):self.stop_patch.stop();self.temp.cleanup()
    def put_rows(self,suite,task,count):
        folder=pe.task_path(self.out,suite,task);folder.mkdir(parents=True,exist_ok=True)
        rows=[dict(task_id=task,episode_idx=i,success=True,num_actions=12) for i in range(count)]
        (folder/'episodes.checkpoint.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    def test_six_workers_resume_without_duplicate_tasks(self):
        self.put_rows(pe.SUITES[0],0,50);self.put_rows(pe.SUITES[0],1,17)
        existing_file=pe.task_path(self.out,pe.SUITES[0],1)/'episodes.checkpoint.jsonl';original=existing_file.read_bytes()
        jobs=queue.Queue()
        for task in pe.pending_tasks(self.out):jobs.put(task)
        self.assertEqual(jobs.qsize(),39)
        stop=threading.Event();pool=pe.ServerPool(6,6,self.ckpt,self.out,'sha',stop,0)
        pool.ensure=lambda slot:f'http://worker{slot}'
        for entry in pool.entries.values():entry['server']=FakeProcess()
        seen=[];mutex=threading.Lock();slots=set()
        def spawn(command,**kwargs):
            suite=command[command.index('--suite')+1];task=int(command[command.index('--task-id')+1])
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'6')
            self.assertEqual(kwargs['env']['MUJOCO_EGL_DEVICE_ID'],'6')
            self.assertEqual(command[command.index('--episodes')+1],'50')
            with mutex:
                self.assertNotIn((suite,task),seen);seen.append((suite,task));slots.add(command[command.index('--url')+1])
            out=Path(command[command.index('--output')+1]);existing=pe.episode_rows(out,task,50)
            with (out/'episodes.checkpoint.jsonl').open('a') as f:
                for i in range(50):
                    if i not in existing:f.write(json.dumps(dict(task_id=task,episode_idx=i,success=True,num_actions=12))+'\n')
            time.sleep(.01);return FakeProcess(0)
        workers=[pe.Worker(slot,pool,stop) for slot in range(6)]
        with patch.object(pe.subprocess,'Popen',side_effect=spawn):
            with ThreadPoolExecutor(max_workers=6) as executor:results=list(executor.map(lambda worker:worker.run(jobs),workers))
        self.assertEqual(results,[[]]*6);self.assertEqual(len(seen),39);self.assertEqual(len(slots),6)
        self.assertEqual(pe.pending_tasks(self.out),[]);self.assertTrue(existing_file.read_bytes().startswith(original))
        self.assertTrue(all(worker.state['tasks_finished']>0 for worker in workers))
        summary=pe.summarize(self.out,50,'sha');self.assertEqual(summary['episodes_completed'],2000)
        pe.evaluation_complete(self.out,'sha')
    def test_six_distinct_model_servers_load_serially_on_gpu6(self):
        stop=threading.Event();pool=pe.ServerPool(6,6,self.ckpt,self.out,'sha',stop,.02)
        mapping={};times=[];loading=[];mutex=threading.Lock()
        def spawn(command,**kwargs):
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'6')
            self.assertEqual(command[command.index('--variant')+1],'A2')
            self.assertEqual(command[command.index('--device')+1],'cuda:0')
            process=FakeProcess();ready=Path(command[command.index('--ready')+1]);ready.write_text(json.dumps(dict(pid=process.pid,port=process.pid)))
            mapping[process.pid]=ready;return process
        def rpc(url,endpoint,payload):
            with mutex:loading.append(url);self.assertEqual(len(loading),1);start=time.time()
            time.sleep(.02)
            with mutex:loading.remove(url);times.append((start,time.time()))
            return dict(variant='A2',step=2000,model_sha256='sha',physical_gpu='6',local_slots=0,goal_slots=16,
                interaction=False,teacher_loaded=False,uses_future=False,uses_state=False)
        with patch.object(pe.subprocess,'Popen',side_effect=spawn),patch.object(pe,'rpc',side_effect=rpc),contextlib.redirect_stdout(io.StringIO()):
            with ThreadPoolExecutor(max_workers=6) as executor:urls=list(executor.map(pool.ensure,range(6)))
        self.assertEqual(len(set(urls)),6);self.assertEqual(len(set(mapping.values())),6);self.assertEqual(len(times),6)
        for first,second in zip(times,times[1:]):self.assertGreaterEqual(second[0]-first[1],.019)
        self.assertTrue(all(entry['health']['physical_gpu']=='6' for entry in pool.entries.values()))
    def test_main_aggregates_full_protocol_and_rejects_changed_checkpoint(self):
        def complete_worker(worker,jobs):
            while True:
                try:suite,task=jobs.get_nowait()
                except queue.Empty:return []
                self.put_rows(suite,task,50);jobs.task_done()
        args=['--checkpoint',str(self.ckpt),'--protocol','full','--output',str(self.out)]
        with patch.object(pe,'WORK',self.root),patch.object(pe,'load_settings',return_value=dict(workers=6,physical_gpu=6,load_gap_seconds=0)),patch.object(pe,'load_config',return_value={'ablation':'A2'}),patch.object(pe,'completed_stage',return_value={'model_sha256':'sha'}),patch.object(pe,'model_metadata',return_value=dict(config={'ablation':'A2'},extra={})),patch.object(pe,'file_sha256',return_value='rollout'),patch.object(pe.Worker,'run',complete_worker),contextlib.redirect_stdout(io.StringIO()):
            pe.main(args)
            self.assertEqual(pe.read(self.out/'status.json')['phase'],'completed')
            self.assertEqual(pe.read(self.out/'summary.json')['episodes_completed'],2000)
            self.assertEqual(pe.read(self.out/'parallel_plan.json')['physical_gpus'],[6]*6)
            protocol=pe.read(self.out/'protocol.json');protocol['checkpoint_sha256']='different'
            (self.out/'protocol.json').write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError,'checkpoint/protocol'):pe.main(args)
    def test_no_processes_start_until_final_training_checkpoint_is_verified(self):
        args=['--checkpoint',str(self.ckpt),'--protocol','full','--output',str(self.out)]
        with patch.object(pe,'WORK',self.root),patch.object(pe,'load_settings',return_value=dict(workers=6,physical_gpu=6,load_gap_seconds=5)),patch.object(pe,'load_config',return_value={'ablation':'A2'}),patch.object(pe,'file_sha256',return_value='rollout'),patch.object(pe,'completed_stage',side_effect=ValueError('not completed')),patch.object(pe.subprocess,'Popen') as spawn:
            with self.assertRaisesRegex(ValueError,'not completed'):pe.main(args)
            spawn.assert_not_called()
    def test_live_progress_tolerates_partial_append_but_resume_rejects(self):
        self.put_rows(pe.SUITES[0],0,3);path=pe.task_path(self.out,pe.SUITES[0],0)
        with (path/'episodes.checkpoint.jsonl').open('a') as f:f.write('{"unfinished":')
        self.assertEqual(len(pe.live_rows(path)),3)
        with self.assertRaises(ValueError):pe.episode_rows(path,0,50)

if __name__=='__main__':unittest.main(verbosity=2)
