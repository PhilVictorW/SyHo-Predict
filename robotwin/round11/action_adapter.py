"""Bounded current-only action-adapter pilot, initialized from the Round10 recipe."""
import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import default_collate
from goal_local.model import GoalLocalPolicy
from goal_local.optim import MasterAdamW
from robotwin.data import CAMERAS, read_frames
from robotwin.round6.model import RetainedTeacher, FutureActionDataset
from robotwin.round6.train import atomic_json, physical_metrics
from robotwin.round7.model import CurrentQueryStudent, OBSERVATION_KEYS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'robotwin/round11/action_adapter_200'
PILOT = ROOT / 'robotwin/round11/paired_query_pilot_500'
KEYS = ['position_cm', 'rotation_deg', 'gripper_mae']
STEPS = 200


class CurrentDataset(FutureActionDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, decode=False)

    def sample(self, episode, t):
        b = super().sample(episode, t)
        images = []
        for cam in CAMERAS:
            pre = 'videos/' + cam
            path = self.root / self.info['video_path'].format(
                video_key=cam, chunk_index=episode[pre+'/chunk_index'],
                file_index=episode[pre+'/file_index'])
            timestamp = float(self.timestamps[episode['dataset_from_index']+t]
                              + episode[pre+'/from_timestamp'])
            images.append(torch.from_numpy(read_frames(path, [timestamp])[0].copy()).permute(2, 0, 1))
        b['current_main'] = images[0]
        b['current_wrist'] = torch.stack(images[1:])
        return b


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'manifest.json').exists() and not args.resume:
        raise RuntimeError('Existing run requires --resume; refusing overwrite')
    state = dict(pid=os.getpid(), phase='loading', physical_gpu=2, step=0,
                 total_steps=STEPS, started_at=time.time(), online_evaluation=False)

    def update(**kw):
        state.update(kw)
        state['updated_at'] = time.time()
        atomic_json(OUT/'status.json', state)

    def record(row):
        row['at'] = time.time()
        with (OUT/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)

    def ac():
        return torch.autocast('cuda', dtype=torch.bfloat16)

    update()
    try:
        assert shutil.disk_usage(OUT).free > 24*2**30
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(.25)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(281114)
        cfg = json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
        recipe = json.loads((ROOT/'robotwin/round7/candidate_recipe.json').read_text())
        teacher = RetainedTeacher(cfg)
        for path in [recipe['source_checkpoint'], recipe['teacher_checkpoint']]:
            saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            assert saved['manifest']['base_config'] == cfg
            named = dict(teacher.named_parameters())
            with torch.no_grad():
                for name, value in saved['adapter'].items():
                    named[name].copy_(value)
            del saved, named
        teacher = teacher.cuda().eval().requires_grad_(False)
        teacher.policy.checkpoint_blocks = False
        teacher.flow.checkpoint_blocks = True
        student = CurrentQueryStudent(teacher).cuda().eval().requires_grad_(False)
        init = torch.load(recipe['initializers'], map_location='cpu', weights_only=False)
        student.set_queries(*init['queries']['fixed_queries'])
        student.gate.load_state_dict(init['gate'])
        del init
        assert not student.gate.scales.any()
        # Train only the new role conditioning adapters; original actor remains frozen.
        params = {}
        for name, p in teacher.named_parameters():
            if name.startswith(('policy.flow.goal_', 'policy.flow.local_')):
                p.requires_grad_(True)
                params[name] = p
        assert params and not teacher.flow.plan.requires_grad and not teacher.flow.residual_gate.requires_grad
        frozen = {n:p._version for n,p in teacher.named_parameters() if n not in params}
        student_versions = {n:p._version for n,p in student.named_parameters()}

        def frozen_check():
            assert all(p._version == frozen[n] and p.grad is None
                       for n,p in teacher.named_parameters() if n in frozen)
            assert all(p._version == student_versions[n] and p.grad is None
                       for n,p in student.named_parameters())

        def forbidden(*a, **kw):
            raise AssertionError('Posterior future input is forbidden in this deployed student pilot')

        def conditions(batch):
            with torch.no_grad(), patch.object(teacher.policy, 'encode_future_images', side_effect=forbidden), \
                 patch.object(teacher.policy.future, 'forward', side_effect=forbidden):
                prep = student.prepare({k:batch[k] for k in OBSERVATION_KEYS})
                enc = GoalLocalPolicy.language(student, prep['current'], student.q_goal[None], student.q_local[None])
            # Keep the role projections outside no_grad: they are part of the trained adapter.
            return (*prep['base'], teacher.flow.goal_proj(enc['goal']),
                    teacher.flow.local_proj(enc['local']), torch.ones(1, 2, device='cuda', dtype=torch.bool))

        train = CurrentDataset(cfg['dataset_root'], teacher.policy.statistics, 'train')
        val = CurrentDataset(cfg['dataset_root'], teacher.policy.statistics, 'val')
        assert set(e['episode_index'] for e in train.selected).isdisjoint(e['episode_index'] for e in val.selected)
        old = json.loads((PILOT/'manifest.json').read_text())
        indices = old['train_indices'][:STEPS*4]
        vi = old['validation_indices']
        source_files = [Path(__file__), ROOT/'robotwin/round7/model.py', ROOT/'robotwin/round5/model.py',
                        ROOT/'robotwin/round3/model.py', ROOT/'robotwin/round2/model.py', ROOT/'goal_local/action.py']
        manifest = dict(kind='current-only small action-adapter pilot', base_config=cfg, recipe=recipe,
            total_steps=STEPS, effective_batch=4, learning_rate=1e-5, warmup_steps=20, save_every=50,
            objective='action flow matching only; no posterior KD or auxiliary future objective',
            trainable_names=list(params), trainable_count=sum(p.numel() for p in params.values()),
            frozen='original Qwen/DINO/LaWM/DiT; future resampler; fixed queries; relevance gate; plan/residual gate',
            initialization='Round10 compatible fixed-query recipe; NOT Round11 trained queries or old StageII3000',
            train_indices=indices, validation_indices=vi, seed=281114,
            action_noise='280920 + 4*zero_based_step + micro; same first200 train anchors as query pilot',
            validation='old16 diagnostic anchors only; not independent confirmation or online success',
            regression_guard='any physical error >1.10 times step0 at two consecutive checks',
            tail_padding='terminal absolute action hold; physical metrics use observed timesteps only',
            decoder='current three views only', student_inputs=list(OBSERVATION_KEYS),
            source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
        if args.resume:
            assert args.resume.resolve().is_relative_to(OUT)
            assert (args.resume.parent/'checkpoint_complete.json').exists()
            assert json.loads((OUT/'manifest.json').read_text()) == manifest
        else:
            atomic_json(OUT/'manifest.json', manifest)
        update(phase='decoding_validation')
        batches = [default_collate([val[i]]) for i in vi]
        optim = MasterAdamW(params.values(), lr=1e-5, weight_decay=1e-8, offload=True)
        baseline = None
        regressions = 0

        @torch.no_grad()
        def evaluate(step):
            teacher.flow.eval()
            update(phase='validation', step=step)
            rows = []
            with torch.random.fork_rng(devices=[0]), ac():
                for i,b in enumerate(batches):
                    update(validation_sample=i+1, validation_total=len(batches))
                    cond = conditions(b)
                    torch.manual_seed(800+i)
                    pred = teacher.flow.sample(cond, torch.randn(1, 50, 32, device='cuda'))
                    rows.append(dict(sample=i, episode=int(b['episode_id'][0]), t=int(b['t'][0]),
                                     metrics=physical_metrics(pred, b, teacher.policy.statistics)))
            means = {k:float(np.mean([r['metrics'][k] for r in rows])) for k in KEYS}
            atomic_json(OUT/f'validation_{step:06d}.json', dict(step=step, rows=rows, mean=means))
            record(dict(step=step, split='validation', mean=means))
            frozen_check()
            return means

        def save(step):
            assert shutil.disk_usage(OUT).free > 16*2**30
            update(phase='saving', step=step)
            dest = OUT/f'step_{step:06d}'
            dest.mkdir(exist_ok=True)
            saved = dict(format='robotwin_current_action_adapter_v1', step=step, manifest=manifest,
                         adapter={n:p.detach().cpu() for n,p in params.items()},
                         optimizer=optim.state_dict(), rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state(), baseline=baseline, regressions=regressions)
            torch.save(saved, dest/'adapter.tmp')
            (dest/'adapter.tmp').replace(dest/'adapter.pt')
            atomic_json(dest/'checkpoint_complete.json', dict(step=step, contains_optimizer=True))

        if not args.resume:
            baseline = evaluate(0)
            expected = json.loads((PILOT/'validation_000000.json').read_text())['mean']['query_only']
            error = max(abs(baseline[k]-expected[k]) for k in KEYS)
            assert error < 1e-6, error
            atomic_json(OUT/'preflight.json', dict(passed=True, initial_reproduction_max_abs=error,
                posterior_access_guard=True, train_validation_disjoint=True))
            save(0)
            start = 0
        else:
            saved = torch.load(args.resume, map_location='cpu', weights_only=False)
            assert saved['format'] == 'robotwin_current_action_adapter_v1' and saved['manifest'] == manifest
            assert set(saved['adapter']) == set(params)
            with torch.no_grad():
                for name,p in params.items(): p.copy_(saved['adapter'][name])
            optim.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['rng'])
            torch.cuda.set_rng_state(saved['cuda_rng'])
            baseline, regressions, start = saved['baseline'], saved['regressions'], saved['step']
            del saved
            # A checkpoint is saved after its validation/guard has completed.
            assert (OUT/f'validation_{start:06d}.json').exists()
            record(dict(step=start, split='resume', checkpoint=str(args.resume)))
        if regressions >= 2:
            update(phase='stopped_by_regression_guard', step=start)
            return
        for step in range(start, STEPS):
            begin = time.time()
            teacher.flow.train()
            teacher.flow.DiT.eval()
            optim.zero_grad()
            loss_value = 0.
            anchors = []
            for micro in range(4):
                update(phase='training', step=step, micro=micro+1)
                batch = default_collate([train[indices[step*4+micro]]])
                anchors.append(dict(episode=int(batch['episode_id'][0]), t=int(batch['t'][0])))
                with ac():
                    cond = conditions(batch)
                    torch.manual_seed(280920+step*4+micro)
                    loss = teacher.flow.loss(batch['actions'].cuda(), batch['actions_mask'].cuda(),
                        batch['action_hz'].cuda(), batch['embodiment_id'].cuda(),
                        batch['remaining_actions'].cuda(), cond, repeats=2)/4
                assert torch.isfinite(loss), 'Nonfinite action loss'
                loss.backward()
                loss_value += float(loss.detach())
                del batch, cond, loss
            grads = [p.grad for p in params.values() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            norm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1., error_if_nonfinite=True))
            assert norm > 0, 'No adapter gradient'
            lr = 1e-5*min(1., (step+1)/20)
            for group in optim.optimizer.param_groups: group['lr'] = lr
            optim.step()
            frozen_check()
            record(dict(step=step+1, split='train', action_loss=loss_value, grad_norm=norm,
                        lr=lr, seconds=time.time()-begin, anchors=anchors))
            update(step=step+1, micro=0)
            if (step+1)%50 == 0:
                measured = evaluate(step+1)
                ratios = {k:measured[k]/max(baseline[k], 1e-10) for k in KEYS}
                regressions = regressions+1 if max(ratios.values()) > 1.10 else 0
                atomic_json(OUT/f'guard_{step+1:06d}.json', dict(ratios=ratios, consecutive_regressions=regressions))
                save(step+1)
                if regressions >= 2:
                    update(phase='stopped_by_regression_guard', step=step+1)
                    return
        atomic_json(OUT/'frozen_audit.json', dict(passed=True, frozen_tensors=len(frozen),
                    student_tensors_frozen=len(student_versions), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        update(phase='completed', step=STEPS, next_phase='Review diagnostic endpoint before a new independent confirmation; no automatic long training')
    except BaseException as exc:
        update(phase='failed', error=f'{type(exc).__name__}: {exc}')
        (OUT/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
