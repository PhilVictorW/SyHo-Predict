"""Episode-local, three-view RoboTwin EEF samples; source files are read-only."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

WORK = Path(__file__).resolve().parent
CACHE = WORK / 'cache'
CAMERAS = ('observation.images.cam_high', 'observation.images.cam_left_wrist',
           'observation.images.cam_right_wrist')
DELTAS = np.array([8, 24, 40, 56, 72, 88])


class ActionNormalizer:
    """Match RobotwinEEFDataConfig: positions min/max, raw quats, inverted grippers."""
    def __init__(self, statistics):
        self.statistics = statistics
        s = statistics['agilex']['action']
        self.low = torch.tensor(s['min'], dtype=torch.float32)
        self.high = torch.tensor(s['max'], dtype=torch.float32)
        assert self.low.shape == self.high.shape == (16,)
        self.scaled = [0, 1, 2, 7, 8, 9, 10, 15]

    def normalize(self, raw):
        x = torch.as_tensor(raw, dtype=torch.float32).clone()
        if x.shape[-1] != 16: raise ValueError('RoboTwin EEF requires 16 dimensions')
        lo, hi = self.low.to(x.device), self.high.to(x.device)
        ids = self.scaled
        x[..., ids] = 2 * (x[..., ids] - lo[ids]) / (hi[ids] - lo[ids]) - 1
        x[..., [7, 15]] *= -1
        return x

    def denormalize(self, normalized):
        x = normalized.float().clone()
        x[..., [7, 15]] *= -1
        lo, hi = self.low.to(x.device), self.high.to(x.device)
        ids = self.scaled
        x[..., ids] = (x[..., ids] + 1) * .5 * (hi[ids] - lo[ids]) + lo[ids]
        return x


def prepare_index(root):
    import fcntl
    root = Path(root).resolve()
    files = sorted((root / 'data').rglob('*.parquet'))
    episode_files = sorted((root / 'meta/episodes').rglob('*.parquet'))
    identity = [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
                for p in files + episode_files + [root / 'meta/info.json']]
    digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    CACHE.mkdir(parents=True, exist_ok=True)
    marker = CACHE / 'index.json'
    with (CACHE / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            result = json.loads(marker.read_text())
            if result['source'] != str(root) or result['fingerprint'] != digest:
                raise ValueError('Source index changed; refusing stale cache')
            return result
        info = json.loads((root / 'meta/info.json').read_text())
        assert info['fps'] == 30 and info['features']['action']['shape'] == [16]
        cols = ['episode_index', 'dataset_from_index', 'dataset_to_index', 'length', 'tasks']
        cols += [f'videos/{cam}/{k}' for cam in CAMERAS for k in
                 ('chunk_index', 'file_index', 'from_timestamp', 'to_timestamp')]
        episodes = [e for p in episode_files for e in pq.read_table(p, columns=cols).to_pylist()]
        episodes.sort(key=lambda e: e['episode_index'])
        assert [e['episode_index'] for e in episodes] == list(range(info['total_episodes']))
        lengths = np.array([e['length'] for e in episodes])
        starts = np.r_[0, np.cumsum(lengths)[:-1]]
        assert [e['dataset_from_index'] for e in episodes] == starts.tolist()
        assert [e['dataset_to_index'] for e in episodes] == np.cumsum(lengths).tolist()
        assert lengths.sum() == info['total_frames']
        count = info['total_frames']
        arrays = {k: np.lib.format.open_memmap(CACHE / (k + '.tmp.npy'), mode='w+', dtype=dtype, shape=shape)
                  for k, dtype, shape in [('actions', 'float32', (count, 16)), ('timestamps', 'float64', (count,))]}
        cursor = 0
        for p in files:
            t = pq.read_table(p, columns=['action', 'timestamp', 'index', 'episode_index', 'frame_index']).combine_chunks()
            n = t.num_rows
            idx = np.arange(cursor, cursor + n)
            eids = np.searchsorted(starts, idx, side='right') - 1
            assert np.array_equal(t['index'].to_numpy(), idx)
            assert np.array_equal(t['episode_index'].to_numpy(), eids)
            frames = idx - starts[eids]
            assert np.array_equal(t['frame_index'].to_numpy(), frames)
            ts = t['timestamp'].to_numpy()
            assert np.allclose(ts, frames / 30., atol=1e-4)
            a = np.asarray(t['action'].combine_chunks().values).reshape(n, 16)
            assert np.isfinite(a).all()
            arrays['actions'][cursor:cursor+n] = a
            arrays['timestamps'][cursor:cursor+n] = ts
            cursor += n
            print(f'Indexed {p.name}: {cursor}/{count} frames', flush=True)
        assert cursor == count
        for k, a in arrays.items():
            a.flush()
            (CACHE / (k + '.tmp.npy')).replace(CACHE / (k + '.npy'))
        for e in episodes:
            for cam in CAMERAS:
                pre = f'videos/{cam}'
                p = root / info['video_path'].format(video_key=cam, chunk_index=e[pre+'/chunk_index'], file_index=e[pre+'/file_index'])
                assert p.is_file() and p.stat().st_size > 0
        result = dict(source=str(root), fingerprint=digest, frames=count, episodes=episodes,
                      action_hz=30, index_and_timestamps_verified=True)
        tmp = marker.with_suffix('.tmp')
        tmp.write_text(json.dumps(result)); tmp.replace(marker)
        return result


class UnreadableTargetFrame(ValueError):
    pass


def log_decode_event(event):
    target = WORK / 'verification'
    target.mkdir(exist_ok=True)
    with (target / f'decode_events.{os.getpid()}.jsonl').open('a') as stream:
        stream.write(json.dumps(event) + '\n')


def read_frames(path, timestamps):
    """Decode only near requested PTS into RGB; limit AV1 CPU thread count."""
    import av
    requested = np.array(timestamps)
    best = [None] * len(requested)
    distances = np.full(len(requested), np.inf)
    tolerance = .018
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 2
        container.seek(max(0, int((requested.min()-.1)/float(stream.time_base))), stream=stream, backward=True)
        bad_packets = []
        done = False
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                bad_packets.append(float(packet.pts * stream.time_base) if packet.pts is not None else None)
                continue
            for frame in frames:
                if frame.pts is None: continue
                at = float(frame.pts * stream.time_base)
                delta = np.abs(requested-at)
                use = np.flatnonzero((delta <= tolerance) & (delta < distances))
                if len(use):
                    rgb = np.asarray(Image.fromarray(frame.to_ndarray(format='rgb24')).resize((256, 256), Image.Resampling.BILINEAR)).copy()
                    for i in use: best[i], distances[i] = rgb, delta[i]
                if at > requested.max()+tolerance:
                    done = True
                    break
            if done: break
    missing = any(x is None for x in best)
    if bad_packets:
        log_decode_event(dict(kind='invalid_av1_packets',path=str(path),packet_times=bad_packets,
                              requested=timestamps,all_target_frames_available=not missing))
    if missing: raise UnreadableTargetFrame(f'Missing exact video frames: {path}, times={timestamps}, distances={distances}')
    return best


class FutureDataset(Dataset):
    def __init__(self, root, statistics, split='train', seed=2026, val_ratio=.05, decode=True):
        if split not in ('train', 'val', 'all'): raise ValueError(split)
        self.root = Path(root)
        meta = prepare_index(root)
        self.episodes, self.fingerprint = meta['episodes'], meta['fingerprint']
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        self.normalizer = ActionNormalizer(statistics)
        self.decode, self.split = decode, split
        # Most language strings occur in a single episode: do NOT put every singleton
        # into validation as the original LIBERO per-instruction split would do.
        rng = np.random.default_rng(seed)
        self.validation_episodes = sorted(rng.permutation(len(self.episodes))[:round(len(self.episodes)*val_ratio)].tolist())
        valid = set(self.validation_episodes)
        self.selected = [e for e in self.episodes if split == 'all' or ((e['episode_index'] in valid) == (split == 'val'))]
        self.offsets = np.cumsum([0] + [e['length'] for e in self.selected])
        self._open()

    def _open(self):
        self.actions = np.load(CACHE / 'actions.npy', mmap_mode='r')
        self.timestamps = np.load(CACHE / 'timestamps.npy', mmap_mode='r')

    def __getstate__(self):
        return {k:v for k,v in self.__dict__.items() if k not in ('actions', 'timestamps')}

    def __setstate__(self, state):
        self.__dict__.update(state); self._open()

    def __len__(self): return int(self.offsets[-1])

    def __getitem__(self, idx):
        if not 0 <= idx < len(self): raise IndexError(idx)
        ei = int(np.searchsorted(self.offsets, idx, side='right')-1)
        episode = self.selected[ei]
        original = int(idx-self.offsets[ei])
        for attempt in range(min(32,episode['length'])):
            t = (original+attempt) % episode['length']
            try:
                sample = self.sample(episode,t)
            except UnreadableTargetFrame as exc:
                log_decode_event(dict(kind='unreadable_sample',episode=episode['episode_index'],t=t,error=str(exc)))
                continue
            if attempt:
                log_decode_event(dict(kind='resampled_anchor',split=self.split,index=int(idx),episode=episode['episode_index'],original_t=original,replacement_t=t))
            return sample
        raise UnreadableTargetFrame(f'No readable anchor after 32 attempts in episode {episode["episode_index"]}')

    def sample(self, e, t):
        length, start = e['length'], e['dataset_from_index']
        if not 0 <= t < length: raise IndexError(t)
        future = t + DELTAS
        valid = future < length
        decoded = {}
        if self.decode:
            for vi, cam in enumerate(CAMERAS):
                steps = [t] + ([int(future[0])] if valid[0] else [])
                if vi == 0: steps += future[1:][valid[1:]].tolist()
                pre = 'videos/' + cam
                path = self.root / self.info['video_path'].format(video_key=cam, chunk_index=e[pre+'/chunk_index'], file_index=e[pre+'/file_index'])
                times = self.timestamps[start+np.array(steps)] + e[pre+'/from_timestamp']
                decoded[vi] = dict(zip(steps, read_frames(path, times.tolist())))
        def image(vi, step, is_valid=True):
            if not self.decode or not is_valid: return torch.zeros(3,256,256,dtype=torch.uint8)
            return torch.from_numpy(decoded[vi][int(step)].copy()).permute(2,0,1)
        remain = min(8, length-t)
        raw = torch.zeros(8,16)
        raw[:remain] = torch.from_numpy(self.actions[start+t:start+t+remain].copy())
        actions, mask = torch.zeros(50,32), torch.zeros(50,32,dtype=torch.bool)
        actions[:remain,:16] = self.normalizer.normalize(raw[:remain]); mask[:remain,:16] = True
        return dict(current_main=image(0,t), current_wrist=torch.stack([image(v,t) for v in (1,2)]),
                    local_main=image(0,future[0],valid[0]),
                    local_wrist=torch.stack([image(v,future[0],valid[0]) for v in (1,2)]),
                    goal_main=torch.stack([image(0,j,v) for j,v in zip(future[1:],valid[1:])]),
                    local_valid=torch.tensor(bool(valid[0])), goal_frame_valid=torch.tensor(valid[1:]),
                    action_chunk=raw, actions=actions, actions_mask=mask,
                    remaining_actions=torch.tensor(length-t), instruction=e['tasks'][0],
                    episode_id=e['episode_index'], t=t, future_indices=torch.tensor(future),
                    action_hz=torch.tensor(30.), embodiment_id=torch.tensor(1))

    def audit(self):
        lengths = np.array([e['length'] for e in self.episodes])
        return dict(episodes=len(lengths), frames=int(lengths.sum()), source=str(self.root),
                    fingerprint=self.fingerprint, index_and_timestamps_verified=True,
                    action_hz=30, action_dimensions=16, embodiment_id=1, cameras=CAMERAS,
                    horizon=8, horizon_seconds=8/30, future_offsets=DELTAS.tolist(),
                    split_method='seeded random episode-level 95/5, never split frames of an episode',
                    validation_episodes=self.validation_episodes, selected_frames=len(self), split=self.split,
                    future_valid_fraction=(np.maximum(lengths[:,None]-DELTAS,0).sum(0)/lengths.sum()).tolist())
