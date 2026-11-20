"""Dataset contract and real Stage-I gradient checks; never modifies source data."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, default_collate

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK.parent))
from indexed_future import IndexedFutureDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    cfg = json.loads((WORK/'configs/stage_I.json').read_text())
    stats = json.loads(Path(cfg['statistics_path']).read_text())
    out = WORK/'verification'
    out.mkdir(exist_ok=True)
    report = {}
    def record(key, value):
        report[key] = value
        print(key, json.dumps(value), flush=True)
        path = out/('gpu.json' if args.gpu else 'data.json')
        path.write_text(json.dumps(report, indent=2))
    ds = IndexedFutureDataset(cfg['dataset_root'], stats, 'train', cfg['seed'], cfg['val_ratio'])
    val = IndexedFutureDataset(cfg['dataset_root'], stats, 'val', cfg['seed'], cfg['val_ratio'])
    assert set(e['episode_index'] for e in ds.selected).isdisjoint(e['episode_index'] for e in val.selected)
    assert len(ds)+len(val) == 2238036
    if not args.gpu:
        audit = ds.audit()
        audit.pop('validation_episodes')
        audit.update(train_episodes=len(ds.selected), val_episodes=len(val.selected), val_frames=len(val))
        record('audit', audit)
        # Exact episode-tail behavior and a deterministic split across fresh instances.
        episode = ds.selected[0]
        for t in [0, episode['length']-9, episode['length']-8, episode['length']-1]:
            s = ds.sample(episode,t)
            remain = min(8,episode['length']-t)
            assert s['actions_mask'].sum() == remain*7
            np.testing.assert_array_equal(s['action_chunk'][:remain].numpy(),ds.actions[episode['dataset_from_index']+t:episode['dataset_from_index']+t+remain])
            assert bool(s['local_valid']) == (t+8 < episode['length'])
            if t == episode['length']-1:
                assert not s['goal_frame_valid'].any() and not s['local_main'].any() and not s['goal_main'].any()
            assert s['instruction'] == episode['tasks'][0] and 'state' not in s
        # Independent JPEG decode must match the adapter output, including camera identity.
        import io, os
        from PIL import Image
        sample = ds.sample(episode,0)
        with (ds.root/f"shard_{episode['shard']:04d}"/'images.bin').open('rb') as f:
            for view, name in [(0,'current_main'),(1,'current_wrist')]:
                off,size=ds.image_offsets[episode['dataset_from_index']].reshape(2,2)[view]
                raw=os.pread(f.fileno(),int(size),int(off))
                expected=torch.from_numpy(np.array(Image.open(io.BytesIO(raw)).convert('RGB'))).permute(2,0,1)
                assert torch.equal(sample[name],expected)
        record('tail_masks_language_camera_identity', True)
        # Check final shard as well as first shard; all ndarray indices audited above.
        final = ds.sample(ds.episodes[-1],0)
        record('last_shard_sample',dict(episode=final['episode_id'],instruction=final['instruction']))
        loader = DataLoader(Subset(ds,[0,len(ds)-1]),batch_size=2,num_workers=2,multiprocessing_context='spawn')
        batch = next(iter(loader))
        assert batch['goal_main'].shape == (2,5,3,256,256)
        assert batch['actions'].shape == (2,50,32)
        record('spawn_loader', {k:list(v.shape) for k,v in batch.items() if isinstance(v,torch.Tensor)})
        # Inherited min/max encloses the actual Plus actions; no silent clipping required.
        lo=ds.actions[:,:6].min(0);hi=ds.actions[:,:6].max(0)
        assert np.all(lo >= np.asarray(stats['franka']['action']['min'][:6])-1e-6)
        assert np.all(hi <= np.asarray(stats['franka']['action']['max'][:6])+1e-6)
        record('inherited_action_statistics_cover_plus',True)
        record('passed',True)
        return
    from goal_local.model import GoalLocalPolicy
    model = GoalLocalPolicy.from_baseline(cfg).to('cuda:0').train()
    record('initialization',model.initialization_report)
    # This smoke update is discarded. Formal training always reloads the original SFT.
    batch = default_collate([ds.sample(ds.selected[0],0)])
    with torch.autocast('cuda',dtype=torch.bfloat16):
        losses = model(batch)
    assert torch.isfinite(losses['loss'])
    losses['loss'].backward()
    names = ['future.goal.query','future.local.query','future.horizon','future.view',
             'act_query','flow_action_query','flow.goal_proj.weight','flow.local_proj.weight','flow.plan']
    params = dict(model.named_parameters())
    gradients = {n:float(params[n].grad.float().norm()) for n in names}
    assert all(np.isfinite(v) and v>0 for v in gradients.values()),gradients
    assert all(p.grad is None for p in model.qwen.visual.parameters())
    assert all(p.grad is None for p in model.dino.parameters())
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True,foreach=False)
    record('flow_loss',float(losses['flow']))
    record('gradient_norms',gradients)
    record('all_gradients_finite_norm',float(norm))
    record('frozen_visual_and_dino_no_grad',True)
    record('peak_gpu_gib',torch.cuda.max_memory_allocated()/2**30)
    record('passed',True)


if __name__ == '__main__':
    main()
