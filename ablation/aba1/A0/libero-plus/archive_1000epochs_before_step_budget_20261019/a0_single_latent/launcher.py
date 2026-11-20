"""Explicit launch; automatically evaluate the final 1000-epoch checkpoint."""
import argparse,json,os,subprocess
from pathlib import Path
from .config import WORK,ROOT,load_config,run_root
from .preflight import inspect

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(WORK/'config.json'))
    p.add_argument('--gpus',default='0,1,3,7')
    p.add_argument('--output');p.add_argument('--resume')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-updates',type=int,default=12)
    p.add_argument('--execute',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--after-eval',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args();cfg=load_config(a.config);plan=inspect(cfg)
    if a.execute and a.dry_run:p.error('Conflicting execution flags')
    if a.gpus!=','.join(map(str,cfg['physical_gpus'])):p.error('Expected GPUs 0,1,3,7')
    output=Path(a.output).resolve() if a.output else run_root(cfg)
    if a.smoke and not a.output:output=output.with_name(output.name+'_smoke')
    if not output.is_relative_to(WORK/'runs'):p.error('Output must stay under A0/libero-plus/runs')
    cmd=[cfg['model_python'],'-B','-m','torch.distributed.run','--standalone','--nnodes=1',
        '--nproc-per-node=4','-m','a0_single_latent.training','--config',str(Path(a.config).resolve()),
        '--output',str(output),'--workers',str(a.workers)]
    if a.resume:cmd+=['--resume',str(Path(a.resume).resolve())]
    if a.smoke:cmd+=['--smoke','--smoke-updates',str(a.smoke_updates)]
    plan.update(command=cmd,output=str(output),smoke=a.smoke,automatic_full_evaluation=a.after_eval and not a.smoke)
    print(json.dumps(plan,indent=2),flush=True)
    if not a.execute:return
    if plan['free_disk_gib']<60:raise OSError('Need 60 GiB free for atomic checkpoints')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpus,PYTHONPATH=str(WORK)+os.pathsep+str(ROOT),
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',NCCL_DEBUG='WARN')
    subprocess.run(cmd,env=env,cwd=WORK,check=True)
    if a.after_eval and not a.smoke:
        final=json.loads((output/'final.json').read_text())
        subprocess.run([cfg['model_python'],'-B',str(WORK/'evaluate.py'),'--config',str(Path(a.config).resolve()),
            '--checkpoint',final['path'],'--scope','full','--gpus',a.gpus,'--execute'],env=env,cwd=WORK,check=True)

if __name__=='__main__':main()
