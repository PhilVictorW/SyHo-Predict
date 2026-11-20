"""Read-only LeRobot v3 sampling with episode-local future and action validity."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq

MAIN = 'observation.images.image'
WRIST = 'observation.images.wrist_image'
HORIZONS = (3, 5, 7, 9, 11)


def indices(t, length, h=8):
    local = t+h
    goal = np.asarray([t+n*h for n in HORIZONS])
    return local, goal, local < length, goal < length


class ActionNormalizer:
    def __init__(self, statistics):
        self.statistics = statistics
        s = statistics['franka']['action']
        self.low = torch.tensor(s['min'][:6])
        self.high = torch.tensor(s['max'][:6])

    def normalize(self, action):
        x = torch.as_tensor(action, dtype=torch.float32).clone()
        x[..., :6] = 2*(x[..., :6]-self.low)/(self.high-self.low)-1
        x[..., 6] = (x[...,6] > .5).float()
        return x

    def denormalize(self, action):
        x = action.float().clone()
        x[..., :6] = (x[..., :6]+1)/2*(self.high-self.low).to(x.device)+self.low.to(x.device)
        x[...,6] = (x[...,6]>.5).float()
        return x


def read_frames(path, timestamps, tolerance=.026):
    """Seek once and decode requested timestamps; reject wrong/absent frames."""
    import av
    if not timestamps:
        return []
    requested = np.asarray(timestamps, dtype=float)
    frames, times = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        container.seek(max(0, int((requested.min()-.1)/float(stream.time_base))), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None: continue
            at = float(frame.pts*stream.time_base)
            if at >= requested.min()-.1:
                frames.append(frame.to_ndarray(format='rgb24')); times.append(at)
            if at > requested.max()+tolerance: break
    if not frames: raise ValueError(f'No decoded frames: {path} {timestamps}')
    selected = np.abs(np.asarray(times)[:,None]-requested[None]).argmin(0)
    if np.max(np.abs(np.asarray(times)[selected]-requested)) > tolerance:
        raise ValueError(f'Video timestamp mismatch: {path} requested={requested}, actual={np.asarray(times)[selected]}')
    return [frames[i] for i in selected]


class FutureDataset(Dataset):
    def __init__(self, root, statistics, split='train', seed=2026, val_ratio=.05, decode=True):
        self.root = Path(root)
        self.info = json.loads((self.root/'meta/info.json').read_text())
        if self.info['fps'] != 20: raise ValueError('This LIBERO contract requires 20 Hz')
        self.episodes = [r for p in sorted((self.root/'meta/episodes').glob('**/*.parquet')) for r in pq.read_table(p).to_pylist()]
        self.episodes.sort(key=lambda r:r['episode_index'])
        self.rows = pq.read_table(sorted((self.root/'data').glob('**/*.parquet')), columns=['action','timestamp','frame_index','episode_index','index','task_index']).combine_chunks()
        self.actions = np.stack(self.rows['action'].to_pylist()).astype(np.float32)
        self.timestamps = self.rows['timestamp'].to_numpy()
        self.frame_indices = self.rows['frame_index'].to_numpy()
        self.episode_indices = self.rows['episode_index'].to_numpy()
        self.global_indices = self.rows['index'].to_numpy()
        if not np.array_equal(self.global_indices, np.arange(len(self.rows))):
            raise ValueError('Parquet file order must preserve global dataset index')
        self.normalizer = ActionNormalizer(statistics)
        self.decode = decode
        groups = defaultdict(list)
        for e in self.episodes: groups[tuple(e['tasks'])].append(e['episode_index'])
        rng = np.random.default_rng(seed)
        validation = set()
        for group in groups.values():
            shuffled = rng.permutation(group)
            validation.update(shuffled[:max(1, round(len(group)*val_ratio))].tolist())
        self.validation_episodes = sorted(validation)
        self.selected = [e for e in self.episodes if split=='all' or ((e['episode_index'] in validation)==(split=='val'))]
        self.offsets = np.cumsum([0]+[e['length'] for e in self.selected])
        self.split = split

    def audit(self):
        counts = np.zeros(6, dtype=np.int64)
        total = 0
        for e in self.episodes:
            start, end = e['dataset_from_index'], e['dataset_to_index']
            length = e['length']
            assert end-start == length
            assert np.all(self.episode_indices[start:end] == e['episode_index'])
            assert np.array_equal(self.frame_indices[start:end], np.arange(length))
            assert np.allclose(np.diff(self.timestamps[start:end]), .05, atol=1e-4)
            for j, delta in enumerate((8,24,40,56,72,88)): counts[j] += max(0,length-delta)
            total += length
        return dict(episodes=len(self.episodes), frames=total, tasks=len({tuple(e['tasks']) for e in self.episodes}),
                    future_valid_fraction=(counts/total).tolist(), validation_episodes=self.validation_episodes,
                    selected_frames=len(self), split=self.split)

    def __len__(self): return int(self.offsets[-1])

    def __getitem__(self, index):
        ei = int(np.searchsorted(self.offsets, index, side='right')-1)
        return self.sample(self.selected[ei], int(index-self.offsets[ei]))

    def sample(self, episode, t):
        length, start = episode['length'], episode['dataset_from_index']
        if not 0 <= t < length: raise IndexError(t)
        li, gi, lv, gv = indices(t, length)
        main_steps = [t]+([li] if lv else [])+gi[gv].tolist()
        wrist_steps = [t]+([li] if lv else [])
        decoded = {}
        if self.decode:
            for camera, steps in [(MAIN, main_steps),(WRIST,wrist_steps)]:
                prefix = f'videos/{camera}'
                path = self.root/self.info['video_path'].format(video_key=camera, chunk_index=episode[prefix+'/chunk_index'], file_index=episode[prefix+'/file_index'])
                times = self.timestamps[start+np.asarray(steps)] + episode[prefix+'/from_timestamp']
                decoded[camera] = dict(zip(steps, read_frames(path, times.tolist())))
        blank = np.zeros((256,256,3), dtype=np.uint8)
        def image(camera, step, valid=True):
            array = decoded[camera][step] if self.decode and valid else blank
            return torch.from_numpy(array.copy()).permute(2,0,1)
        remain = min(8, length-t)
        raw = torch.zeros(8,7)
        raw[:remain] = torch.from_numpy(self.actions[start+t:start+t+remain])
        actions, mask = torch.zeros(50,32), torch.zeros(50,32,dtype=torch.bool)
        actions[:remain,:7] = self.normalizer.normalize(raw[:remain]); mask[:remain,:7] = True
        return dict(current_main=image(MAIN,t), current_wrist=image(WRIST,t),
            local_main=image(MAIN,li,lv), local_wrist=image(WRIST,li,lv),
            goal_main=torch.stack([image(MAIN,int(j),bool(v)) for j,v in zip(gi,gv)]),
            local_valid=torch.tensor(lv), goal_frame_valid=torch.tensor(gv), action_chunk=raw,
            actions=actions, actions_mask=mask, remaining_actions=torch.tensor(length-t),
            instruction=episode['tasks'][0], episode_id=episode['episode_index'], t=t,
            future_indices=torch.tensor([li]+gi.tolist()), action_hz=torch.tensor(20.), embodiment_id=torch.tensor(25))
