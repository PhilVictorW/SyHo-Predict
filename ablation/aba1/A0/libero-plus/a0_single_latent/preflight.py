import argparse,json,math,shutil
from pathlib import Path
from .config import WORK,ROOT,load_config,run_root

def inspect(cfg):
    for key in ('baseline_checkpoint','qwen_path','dino_path','statistics_path','dataset_root','full_eval_manifest','benchmark','model_python','simulator_python'):
        if not Path(cfg[key]).exists():raise FileNotFoundError(cfg[key])
    manifest=json.loads((Path(cfg['dataset_root'])/'manifest.json').read_text())
    if (manifest['frames'],manifest['episodes'])!=(2238036,14347):raise ValueError('Dataset changed')
    meta=json.loads((ROOT/'libero-plus/cache/index/metadata.json').read_text())
    if meta['source']!=cfg['dataset_root']:raise ValueError('Wrong read-only index')
    n=2125407;batch=cfg['world_size']*cfg['micro_batch']*cfg['accumulation']
    return dict(architecture=cfg['architecture'],condition='latent16 -> 768 + current DINO -> original DiT',
        uses_dino=True,uses_teacher=False,uses_future=False,loss='action_flow_only',
        dataset=cfg['dataset_root'],train_anchors=n,validation_anchors=112629,budget_unit='optimizer_updates',
        updates_per_epoch=math.ceil(n/batch),total_updates=cfg['stop_after_updates'],schedule_total_steps=cfg['schedule_total_steps'],
        total_training_anchors=cfg['stop_after_updates']*batch,nominal_epochs=cfg['stop_after_updates']*batch/n,global_batch=batch,physical_gpus=cfg['physical_gpus'],
        processes_per_gpu=1,distributed='FSDP FP32 shards / BF16 compute',
        output=str(run_root(cfg)),free_disk_gib=shutil.disk_usage(WORK).free/2**30,training_started=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(WORK/'config.json'));a=p.parse_args()
    print(json.dumps(inspect(load_config(a.config)),indent=2))

if __name__=='__main__':main()
