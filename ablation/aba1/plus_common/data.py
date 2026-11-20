"""Selective read-only indexed RLDS. Never builds/overwrites the historical cache."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
from goal_local.data import ActionNormalizer
from .config import ROOT

def _load_indexed():
    name="aba1_plus_readonly_indexed"
    if name in sys.modules: return sys.modules[name]
    path=ROOT/"libero-plus/runtime/indexed_future.py"
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

upstream=_load_indexed()

class IndexedFutureDataset(upstream.IndexedFutureDataset):
    def __init__(self,root,statistics,split="train",seed=2026,val_ratio=.05,decode=True,
                 local=True,goal=True):
        for name in ("metadata.json","actions.npy","offsets.npy"):
            if not (upstream.CACHE/name).is_file():
                raise FileNotFoundError(f"Read-only index missing: {upstream.CACHE/name}")
        self.use_local,self.use_goal=bool(local),bool(goal)
        super().__init__(root,statistics,split,seed,val_ratio,decode)

    def sample(self,episode,t):
        length,start=episode["length"],episode["dataset_from_index"]
        if not 0<=t<length: raise IndexError(t)
        remain=min(8,length-t)
        raw=torch.zeros(8,7)
        raw[:remain]=torch.from_numpy(self.actions[start+t:start+t+remain].copy())
        actions=torch.zeros(50,32);mask=torch.zeros(50,32,dtype=torch.bool)
        actions[:remain,:7]=self.normalizer.normalize(raw[:remain]);mask[:remain,:7]=True
        out=dict(current_main=self._image(episode,t,0),current_wrist=self._image(episode,t,1),
            action_chunk=raw,actions=actions,actions_mask=mask,remaining_actions=torch.tensor(length-t),
            instruction=episode["tasks"][0],episode_id=episode["episode_index"],t=t,
            action_hz=torch.tensor(20.),embodiment_id=torch.tensor(25))
        if self.use_local:
            li=t+8;lv=li<length
            out.update(local_main=self._image(episode,li,0,lv),
                       local_wrist=self._image(episode,li,1,lv),local_valid=torch.tensor(lv))
        if self.use_goal:
            gi=[t+k for k in (24,40,56,72,88)];gv=[j<length for j in gi]
            out.update(goal_main=torch.stack([self._image(episode,j,0,v) for j,v in zip(gi,gv)]),
                       goal_frame_valid=torch.tensor(gv,dtype=torch.bool))
        return out
