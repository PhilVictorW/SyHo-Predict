import json,tempfile,torch
from pathlib import Path
from torch.utils.data import DataLoader
from ablation.aba1.plus_common.data import IndexedFutureDataset
if __name__=="__main__":
 torch.set_num_threads(1)
 cfg=json.loads(Path("config.json").read_text())
 ds=IndexedFutureDataset(cfg["dataset_root"],json.loads(Path(cfg["statistics_path"]).read_text()),"train",2026,local=True,goal=False)
 loader=DataLoader(ds,batch_size=1,num_workers=1,multiprocessing_context="spawn",prefetch_factor=1)
 it=iter(loader);batch=next(it)
 assert "local_main" in batch and "goal_main" not in batch
 assert batch["current_main"].shape==(1,3,256,256)
 print("Real Local-only DataLoader IPC passed; tmp="+tempfile.gettempdir(),flush=True)
 del it;ds.close()
