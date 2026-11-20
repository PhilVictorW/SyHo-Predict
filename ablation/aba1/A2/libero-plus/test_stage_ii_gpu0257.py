"""Four-card admission constraints and uneven five-rank synchronization."""
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch.multiprocessing as mp
from a2_stage2 import launch, runtime
from test_adaptive_stage_ii import gradient_worker

WORK = Path(__file__).resolve().parent

class Hybrid(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((WORK/'config.json').read_text())
        self.children=[];self.started=[];self.observed=[]
        self.gpu5_budget=45000
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.record=self.stack.enter_context(patch.object(launch,'record'))
        self.stack.enter_context(patch.object(launch,'resources_all',side_effect=self.memory))
        self.stack.enter_context(patch.object(launch,'observe',side_effect=self.observe))
        self.stack.enter_context(patch.object(launch,'start_candidate',side_effect=self.start))
        self.stack.enter_context(patch.object(launch,'terminate'))
        self.sleep=self.stack.enter_context(patch.object(launch.time,'sleep'))

    def memory(self):
        return {gpu:dict(physical_gpu=gpu,gpu_free_mib=(self.gpu5_budget if gpu==5 else 80000)
                         -sum(child.gpu==gpu for child in self.children)*10000,
                         gpu_total_mib=81920,host_available_mib=100000) for gpu in (0,2,5,7)}

    def observe(self,children,profiles):
        self.observed.append(len(profiles));return self.memory()

    def start(self,cfg,control,rank,children,resume,gpu):
        self.started.append((rank,gpu))
        children.append(SimpleNamespace(gpu=gpu,poll=lambda:None))
        return dict(rank=rank,physical_gpu=gpu,process_gpu_mib=10000,
                    host_peak_mib=1000,optimizer_headroom_mib=8000)

    def run_admission(self,**kwargs):
        return launch.admit_hybrid(self.cfg,WORK,self.children,**kwargs)

    def test_other_cards_once_and_only_gpu5_grows(self):
        profiles=self.run_admission()
        self.assertEqual([p['physical_gpu'] for p in profiles],[0,2,7,5,5,5])
        self.assertEqual(self.started,list(enumerate([0,2,7,5,5,5])))
        self.assertEqual(self.observed,list(range(1,7))+[6])
        spec=runtime.selected_spec(self.cfg,len(profiles),[p['physical_gpu'] for p in profiles])
        self.assertEqual(sum(spec['accumulation_per_rank']),32)
        self.assertEqual(spec['save_every'],100)
        self.assertEqual(spec['stop_after_updates'],2500)

    def test_required_gpu_waits_instead_of_reducing_or_moving_rank(self):
        self.gpu5_budget=20000
        available=[False]
        original=self.memory
        def memory():
            rows=original()
            if not available[0]:rows[2]['gpu_free_mib']=1000
            return rows
        self.sleep.side_effect=lambda seconds:available.__setitem__(0,True)
        with patch.object(launch,'resources_all',side_effect=memory):
            profiles=self.run_admission()
        self.assertTrue(self.sleep.called)
        self.assertEqual([p['physical_gpu'] for p in profiles],[0,2,7,5])
        self.assertEqual(self.started,list(enumerate([0,2,7,5])))

    def test_optional_candidate_rejection_preserves_all_required_cards(self):
        original=self.start
        def start(*args):
            if args[2]==4:raise runtime.CapacityError('memory changed before CUDA transfer')
            return original(*args)
        with patch.object(launch,'start_candidate',side_effect=start):profiles=self.run_admission()
        self.assertEqual([p['physical_gpu'] for p in profiles],[0,2,7,5])
        self.assertEqual(len(self.children),4)

    def test_required_candidate_retries_same_gpu_and_rank(self):
        self.gpu5_budget=20000;attempts=[];original=self.start
        def start(*args):
            attempts.append((args[2],args[-1]))
            if attempts.count((1,2))==1 and args[2]==1:raise runtime.CapacityError('temporary pressure')
            return original(*args)
        with patch.object(launch,'start_candidate',side_effect=start):profiles=self.run_admission()
        self.assertEqual(attempts[:3],[(0,0),(1,2),(1,2)])
        self.assertEqual([p['physical_gpu'] for p in profiles],[0,2,7,5])

    def test_optional_candidate_withdrawn_when_observed_headroom_falls(self):
        def observe(children,profiles):
            rows=self.memory()
            if len(profiles)==5:rows[5]['gpu_free_mib']=2000
            return rows
        with patch.object(launch,'observe',side_effect=observe):profiles=self.run_admission()
        self.assertEqual([p['physical_gpu'] for p in profiles],[0,2,7,5])
        self.assertEqual(len(self.children),4)

    def test_resume_uses_exact_saved_mapping_without_extra_ranks(self):
        self.gpu5_budget=80000
        mapping=[0,2,7,5,5]
        profiles=self.run_admission(resume=Path('saved'),required_mapping=mapping)
        self.assertEqual([p['physical_gpu'] for p in profiles],mapping)
        self.assertEqual(len(self.children),5)

    def test_selected_spec_rejects_duplicate_fixed_card_and_missing_gpu5(self):
        for mapping in ([0,2,7],[0,2,7,5,0],[0,2,5,7],[0,2,7,5,2]):
            with self.assertRaises(ValueError):runtime.selected_spec(self.cfg,len(mapping),mapping)
        self.assertEqual(runtime.selected_spec(self.cfg,5)['accumulation_per_rank'],[7,7,6,6,6])

    def test_uneven_five_rank_gradient_matches_batch32(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(gradient_worker,args=(str(Path(directory)/'gloo'),5),nprocs=5,join=True)

if __name__=='__main__':unittest.main(verbosity=2)
