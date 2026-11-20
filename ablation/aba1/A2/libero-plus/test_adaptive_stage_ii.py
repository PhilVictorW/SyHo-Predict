"""CPU-only admission, immutable Stage I, and uneven-rank optimizer checks."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from a2_stage2 import launch, runtime
from a2_local.config_runtime import code_identity
from a2_local.distributed import DistributedMaster
from goal_local.optim import MasterAdamW

WORK = Path(__file__).resolve().parent


def gradient_worker(rank, rendezvous, world):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=world)
    try:
        p = torch.nn.Parameter(torch.tensor([.3, -.2], dtype=torch.float32))
        optimizer = DistributedMaster([p], 1e-3, 1e-8, bucket_elements=2)
        optimizer.broadcast_weights()
        expected = torch.nn.Parameter(p.detach().clone())
        reference = MasterAdamW([expected], 1e-3, 1e-8)
        for step in range(2):
            optimizer.zero_grad()
            for i in range(rank, 32, world):
                target = torch.tensor([i / 32, step / 10])
                ((p - target).square().sum() * world / 32).backward()
            optimizer.step(1e-3, 1)
            reference.zero_grad()
            for i in range(32):
                target = torch.tensor([i / 32, step / 10])
                ((expected - target).square().sum() / 32).backward()
            torch.nn.utils.clip_grad_norm_([expected], 1, foreach=False)
            reference.step()
            torch.testing.assert_close(p, expected, rtol=1e-5, atol=1e-7)
    finally:
        dist.destroy_process_group()


class Adaptive(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((WORK / 'config.json').read_text())

    def test_current_stage_I_code_and_config_are_unchanged(self):
        root = WORK / 'protocol_plus_v1/seed_2026/I'
        self.assertEqual(code_identity(), json.loads((root / 'code_identity.json').read_text()))
        self.assertEqual(self.cfg, json.loads((root / 'contract.json').read_text())['config'])

    def test_every_supported_world_preserves_32_samples(self):
        for world in range(4, 33):
            plan = runtime.selected_spec(self.cfg, world)
            self.assertEqual(sum(plan['accumulation_per_rank']), 32)
            self.assertGreaterEqual(min(plan['accumulation_per_rank']), 1)
            self.assertEqual(plan['stop_after_updates'], 2500)
            self.assertEqual(plan['save_every'], 100)
        with self.assertRaises(ValueError):
            runtime.selected_spec(self.cfg, 33)

    def test_admission_needs_both_gpu_and_optimizer_cpu_headroom(self):
        profile = dict(process_gpu_mib=10000, host_peak_mib=16000, optimizer_headroom_mib=30000)
        good = dict(gpu_free_mib=20000, gpu_total_mib=81920, host_available_mib=60000)
        self.assertTrue(runtime.admission(good, [profile])['allowed'])
        self.assertFalse(runtime.admission(dict(good, gpu_free_mib=15000), [profile])['allowed'])
        self.assertFalse(runtime.admission(dict(good, host_available_mib=40000), [profile])['allowed'])

    def test_launch_observe_then_add_until_both_cards_capacity(self):
        children = []; events = []
        profile = dict(process_gpu_mib=10000, host_peak_mib=1000, optimizer_headroom_mib=8000)
        def memory(gpu):
            initial = 45000 if gpu == 5 else 25000
            return dict(physical_gpu=gpu, gpu_free_mib=initial - children.count(gpu) * 10000,
                        gpu_total_mib=81920, host_available_mib=100000)
        def start(cfg, control, rank, running, resume, gpu):
            events.append(('start', rank, gpu)); running.append(gpu)
            return dict(profile, rank=rank, physical_gpu=gpu)
        def observe(running, profiles):
            events.append(('observe', len(profiles))); return {gpu: memory(gpu) for gpu in (5,0)}
        with patch.dict(runtime.POLICY, physical_gpus=[5,0]), patch.object(launch, 'resources', side_effect=memory), patch.object(launch, 'record'), \
             patch.object(launch, 'start_candidate', side_effect=start), patch.object(launch, 'observe', side_effect=observe):
            profiles = launch.admit(self.cfg, WORK, children)
        self.assertEqual([p['physical_gpu'] for p in profiles], [5,0,5,5])
        self.assertEqual(events, [('start',0,5),('observe',1),('start',1,0),('observe',2),
                                  ('start',2,5),('observe',3),('start',3,5),('observe',4)])

    def test_rejected_card_can_continue_on_other_card(self):
        children = []
        memory = {gpu:dict(physical_gpu=gpu,gpu_free_mib=80000,gpu_total_mib=81920,
                           host_available_mib=100000) for gpu in (5,0)}
        def start(cfg, control, rank, running, resume, gpu):
            if gpu == 0 or rank == 2:
                raise runtime.CapacityError('external workload took memory')
            running.append(object())
            return dict(physical_gpu=gpu, process_gpu_mib=10000, host_peak_mib=1000, optimizer_headroom_mib=8000)
        with patch.object(launch, 'resources_all', return_value=memory), patch.object(launch, 'record'), \
             patch.object(launch, 'start_candidate', side_effect=start), patch.object(launch, 'observe', return_value=memory):
            profiles = launch.admit(self.cfg, WORK, children)
        self.assertEqual(len(profiles), 2); self.assertEqual(len(children), 2)
        self.assertEqual([p['physical_gpu'] for p in profiles], [5,5])

    def test_transient_rejected_card_reenters_after_stable_observation(self):
        children=[]; attempts=[]
        def memory(gpu):
            return dict(physical_gpu=gpu,gpu_free_mib=30000 if len(children)<2 else 10000,
                        host_available_mib=100000,gpu_total_mib=81920)
        def start(cfg,control,rank,running,resume,gpu):
            attempts.append(gpu)
            if len(attempts)==1:raise runtime.CapacityError('temporary GPU5 pressure')
            running.append(gpu)
            return dict(physical_gpu=gpu,process_gpu_mib=10000,host_peak_mib=1000,optimizer_headroom_mib=8000)
        with patch.dict(runtime.POLICY, physical_gpus=[5,0]), patch.object(launch,'resources',side_effect=memory), patch.object(launch,'record'), \
             patch.object(launch,'start_candidate',side_effect=start), \
             patch.object(launch,'observe',side_effect=lambda *a:{gpu:memory(gpu) for gpu in (5,0)}):
            profiles=launch.admit(self.cfg,WORK,children)
        self.assertEqual(attempts,[5,0,5])
        self.assertEqual([p['physical_gpu'] for p in profiles],[0,5])

    def test_shared_host_memory_limits_both_cards(self):
        snapshots = {gpu:dict(gpu_free_mib=80000,host_available_mib=40000) for gpu in (5,0)}
        profile = dict(physical_gpu=5,process_gpu_mib=10000,host_peak_mib=16000,optimizer_headroom_mib=30000)
        self.assertIsNone(launch.choose_gpu(snapshots,[profile])[0])

    def test_single_visible_gpu_is_local_zero_on_either_card(self):
        for gpu in (0,2,5,7):
            with patch.dict('os.environ', CUDA_VISIBLE_DEVICES=str(gpu), WORLD_SIZE='8'):
                self.assertEqual(runtime.worker_physical_gpu(),gpu)
                self.assertEqual(runtime.rank_device(self.cfg,7),0)
        with patch.dict('os.environ', CUDA_VISIBLE_DEVICES='5,0'):
            with self.assertRaises(ValueError):runtime.worker_physical_gpu()

    def test_selected_mapping_is_explicit_and_batch_preserved(self):
        for mapping in ([0,2,7,5],[0,2,7,5,5],[0,2,7,5,5,5]):
            spec=runtime.selected_spec(self.cfg,len(mapping),mapping)
            self.assertEqual(spec['rank_to_gpu'],mapping)
            self.assertEqual(sum(spec['accumulation_per_rank']),32)
        for mapping in ([0,2,7], [0,2,7,5,0], [0,2,5,7]):
            with self.assertRaises(ValueError):runtime.selected_spec(self.cfg,len(mapping),mapping)

    def test_resume_selects_the_saved_card_and_checks_all_used_cards(self):
        snapshots={gpu:dict(gpu_free_mib=50000,host_available_mib=100000) for gpu in (5,0)}
        profile=dict(physical_gpu=5,process_gpu_mib=10000,host_peak_mib=16000,optimizer_headroom_mib=30000)
        self.assertEqual(launch.choose_gpu(snapshots,[profile],required_gpu=5)[0],5)
        snapshots[5]['gpu_free_mib']=2000
        self.assertFalse(launch.prefix_safe([profile],snapshots))
        self.assertEqual(launch.choose_gpu(snapshots,[profile])[0],0)

    def test_premature_launch_never_overwrites_stage_I_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = dict(phase='training', stage='I', step=2000)
            runtime.atomic_json(root / 'pipeline_status.json', original)
            with patch.object(launch, 'run_root', return_value=root), \
                 patch.object(launch, 'stage_root', return_value=root / 'II'), \
                 patch.object(launch, 'record'), \
                 patch.object(launch, 'teacher_ready', side_effect=RuntimeError('Stage I is active')), \
                 patch('sys.argv', ['adaptive', '--execute']):
                with self.assertRaises(RuntimeError):
                    launch.main()
            self.assertEqual(runtime.read_json(root / 'pipeline_status.json'), original)

    def test_probe_gradients_do_not_change_weights(self):
        from a2_stage2.probe import ProbeGradients
        p = torch.nn.Parameter(torch.tensor([.3, -.2]))
        original = p.detach().clone(); hooks = ProbeGradients([p])
        (p.square().sum() / 32).backward()
        (p.square().sum() / 32).backward()
        torch.testing.assert_close(p, original)
        torch.testing.assert_close(hooks.grads[0], original / 8)
        self.assertIsNone(p.grad); hooks.close()

    def test_uneven_three_rank_gradient_matches_batch32(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(gradient_worker, args=(str(Path(directory) / 'gloo'), 3), nprocs=3, join=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
