"""Only current dual views and action targets; no future image decoding."""
from ablation.aba1.plus_common.data import IndexedFutureDataset, ActionNormalizer

class CurrentDataset(IndexedFutureDataset):
    def __init__(self,root,statistics,split='train',seed=2026,val_ratio=.05,decode=True):
        super().__init__(root,statistics,split,seed,val_ratio,decode,local=False,goal=False)
