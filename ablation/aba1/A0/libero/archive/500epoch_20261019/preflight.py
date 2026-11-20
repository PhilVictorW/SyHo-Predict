import json,sys,shutil
from pathlib import Path
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK.parents[3]))
from a0_single_latent.config import load_config,atomic_json
if __name__=='__main__':
 c=load_config(WORK/'config.json')
 paths={k:Path(c[k]).exists() for k in ('baseline_checkpoint','qwen_path','statistics_path','dataset_root')}
 report=dict(architecture=c['architecture'],paths=paths,physical_gpus=c['physical_gpus'],world_size=3,global_batch=32,
             epochs=500,expected_updates=4074000,uses_dino=False,disk_free_gib=shutil.disk_usage(WORK).free/1024**3,
             starts_training=False)
 print(json.dumps(report,indent=2));atomic_json(WORK/'preflight_report.json',report)
 if not all(paths.values()):raise SystemExit(1)
