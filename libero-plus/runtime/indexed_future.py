"""Read-only indexed RLDS backend for the existing Goal/Local sample contract."""
import hashlib
import io
import json
import os
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from goal_local.data import ActionNormalizer, indices

WORK = Path(__file__).resolve().parents[1]
CACHE = WORK / 'cache/index'


def fingerprint(root):
    return hashlib.sha256((Path(root) / 'manifest.json').read_bytes()).hexdigest()


def prepare_index(root):
    """Materialize only small numeric indices; source JPEG bytes remain in place."""
    root = Path(root).resolve()
    digest = fingerprint(root)
    marker = CACHE / 'metadata.json'
    if marker.exists():
        meta = json.loads(marker.read_text())
        if meta['source'] != str(root) or meta['manifest_sha256'] != digest:
            raise ValueError('Index source changed; use a separate cache for a different dataset')
        return meta
    import fcntl
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            return prepare_index(root)
        manifest = json.loads((root / 'manifest.json').read_text())
        rows = sorted(manifest['shards'], key=lambda r: r['shard'])
        assert len(rows) == 1024 and [r['shard'] for r in rows] == list(range(1024))
        assert manifest['frames'] == 2238036 and manifest['episodes'] == 14347
        assert manifest['action_hz'] == 20 and manifest['archive_crc_verified']
        actions = np.lib.format.open_memmap(CACHE / 'actions.tmp.npy', mode='w+', dtype=np.float32, shape=(manifest['frames'], 7))
        offsets = np.lib.format.open_memmap(CACHE / 'offsets.tmp.npy', mode='w+', dtype=np.int64, shape=(manifest['frames'], 4))
        episodes, cursor = [], 0
        for row in rows:
            shard = root / f"shard_{row['shard']:04d}"
            assert json.loads((shard / 'complete.json').read_text()) == row
            assert row['archive_crc_verified']
            with np.load(shard / 'arrays.npz', allow_pickle=False) as z:
                a, off, ends = z['actions'], z['offsets'], z['ends']
                lengths = np.diff(np.r_[0, ends])
                assert a.shape == (row['frames'], 7) and off.shape == (row['frames'], 4)
                assert len(ends) == row['episodes'] and ends[-1] == row['frames'] and (lengths > 0).all()
                assert np.array_equal(z['episode'], np.repeat(np.arange(len(ends)), lengths))
                assert np.isfinite(a).all() and set(np.unique(a[:, 6])) <= {0., 1.}
                flat = off.reshape(-1, 2)
                assert flat[0, 0] == 0 and (flat[:, 1] > 0).all()
                assert np.array_equal(flat[:-1].sum(1), flat[1:, 0])
                assert int(flat[-1].sum()) == (shard / 'images.bin').stat().st_size == row['bytes']
                actions[cursor:cursor+len(a)] = a
                offsets[cursor:cursor+len(a)] = off
                start = 0
                for local_id, end in enumerate(ends):
                    episodes.append(dict(episode_index=len(episodes), shard=row['shard'], source_episode=local_id,
                        length=int(end)-start, dataset_from_index=cursor+start, dataset_to_index=cursor+int(end),
                        tasks=[str(z['languages'][local_id])], origin=str(z['origins'][local_id])))
                    start = int(end)
                cursor += len(a)
            if (row['shard']+1) % 256 == 0:
                print(f"Indexed {row['shard']+1}/1024 shards", flush=True)
        assert cursor == manifest['frames'] and len(episodes) == manifest['episodes']
        actions.flush(); offsets.flush()
        del actions, offsets
        (CACHE / 'actions.tmp.npy').replace(CACHE / 'actions.npy')
        (CACHE / 'offsets.tmp.npy').replace(CACHE / 'offsets.npy')
        meta = dict(source=str(root), manifest_sha256=digest, frames=cursor, episodes=episodes,
            action_hz=20, frequency_provenance='source conversion manifest, no per-frame timestamps',
            gripper='already converted: 1=open, 0=closed; do not invert again', image_transform='RGB decode only')
        tmp = CACHE / 'metadata.tmp.json'
        tmp.write_text(json.dumps(meta, ensure_ascii=False))
        tmp.replace(marker)
        return meta


class IndexedFutureDataset(Dataset):
    def __init__(self, root, statistics, split='train', seed=2026, val_ratio=.05, decode=True):
        if split not in ('train', 'val', 'all'):
            raise ValueError(split)
        self.root = Path(root).resolve()
        meta = prepare_index(self.root)
        self.episodes = meta['episodes']
        self.source_digest = meta['manifest_sha256']
        self.normalizer = ActionNormalizer(statistics)
        self.decode, self.split = decode, split
        groups = defaultdict(list)
        for e in self.episodes:
            groups[tuple(e['tasks'])].append(e['episode_index'])
        rng = np.random.default_rng(seed)
        validation = set()
        for group in groups.values():
            validation.update(rng.permutation(group)[:max(1, round(len(group)*val_ratio))].tolist())
        self.validation_episodes = sorted(validation)
        self.selected = [e for e in self.episodes if split == 'all' or ((e['episode_index'] in validation) == (split == 'val'))]
        self.offsets = np.cumsum([0] + [e['length'] for e in self.selected])
        self._open_indices()

    def _open_indices(self):
        self.actions = np.load(CACHE / 'actions.npy', mmap_mode='r')
        self.image_offsets = np.load(CACHE / 'offsets.npy', mmap_mode='r')
        self._fds = OrderedDict()

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in ('actions', 'image_offsets', '_fds')}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_indices()

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ei = int(np.searchsorted(self.offsets, index, side='right') - 1)
        return self.sample(self.selected[ei], int(index-self.offsets[ei]))

    def _image(self, episode, step, view, valid=True):
        if not self.decode or not valid:
            return torch.zeros(3, 256, 256, dtype=torch.uint8)
        assert 0 <= step < episode['length']
        sid = episode['shard']
        if sid not in self._fds:
            fd = os.open(self.root / f'shard_{sid:04d}' / 'images.bin', os.O_RDONLY)
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            self._fds[sid] = fd
            if len(self._fds) > 16:
                _, old = self._fds.popitem(last=False)
                os.close(old)
        self._fds.move_to_end(sid)
        off, size = self.image_offsets[episode['dataset_from_index']+step].reshape(2, 2)[view]
        data = os.pread(self._fds[sid], int(size), int(off))
        if len(data) != size:
            raise IOError(f'Short image read: shard={sid}, step={step}, view={view}')
        rgb = np.array(Image.open(io.BytesIO(data)).convert('RGB'))
        if rgb.shape != (256, 256, 3):
            raise ValueError(f'Unexpected image shape: {rgb.shape}')
        return torch.from_numpy(rgb).permute(2, 0, 1)

    def sample(self, episode, t):
        length, start = episode['length'], episode['dataset_from_index']
        if not 0 <= t < length:
            raise IndexError(t)
        li, gi, lv, gv = indices(t, length)
        remain = min(8, length-t)
        raw = torch.zeros(8, 7)
        raw[:remain] = torch.from_numpy(self.actions[start+t:start+t+remain].copy())
        actions, mask = torch.zeros(50, 32), torch.zeros(50, 32, dtype=torch.bool)
        actions[:remain, :7] = self.normalizer.normalize(raw[:remain])
        mask[:remain, :7] = True
        return dict(current_main=self._image(episode,t,0), current_wrist=self._image(episode,t,1),
            local_main=self._image(episode,li,0,lv), local_wrist=self._image(episode,li,1,lv),
            goal_main=torch.stack([self._image(episode,int(j),0,bool(v)) for j,v in zip(gi,gv)]),
            local_valid=torch.tensor(lv), goal_frame_valid=torch.tensor(gv), action_chunk=raw,
            actions=actions, actions_mask=mask, remaining_actions=torch.tensor(length-t),
            instruction=episode['tasks'][0], episode_id=episode['episode_index'], t=t,
            future_indices=torch.tensor([li]+gi.tolist()), action_hz=torch.tensor(20.), embodiment_id=torch.tensor(25))

    def audit(self):
        lengths = np.asarray([e['length'] for e in self.episodes])
        counts = np.maximum(lengths[:, None]-np.asarray([8,24,40,56,72,88]), 0).sum(0)
        assert sum(lengths) == len(self.actions)
        assert np.isfinite(self.actions).all()
        return dict(episodes=len(lengths), frames=len(self.actions), tasks=len({tuple(e['tasks']) for e in self.episodes}),
            future_valid_fraction=(counts/len(self.actions)).tolist(), validation_episodes=self.validation_episodes,
            selected_frames=len(self), split=self.split, source=str(self.root), source_manifest_sha256=self.source_digest,
            backend='indexed_rlds', action_hz=20, timestamps_available=False,
            state_used=False, gripper_already_converted=True, image_reencoding=False)
