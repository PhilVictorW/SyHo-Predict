"""Finish the active two-GPU validation, then run 1000 updates and full evaluation."""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[3]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validation-launcher-pid',type=int,required=True);a=parser.parse_args()
    cfg=json.loads((WORK/'config.json').read_text());run=WORK/'runs'/cfg['run_name'];validation=run.with_name(run.name+'_smoke')
    state_file=WORK/'workflow_step1000_gpu07.json'
    def state(phase,**kw):
        data=dict(phase=phase,pid=os.getpid(),time=time.time(),gpus=cfg['physical_gpus'],stop_after_updates=cfg['stop_after_updates'],run=str(run),validation=str(validation),**kw)
        temp=state_file.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(state_file)
        print(json.dumps(data),flush=True)
    try:
        state('waiting_for_validation',validation_launcher_pid=a.validation_launcher_pid)
        while True:
            proc=Path('/proc')/str(a.validation_launcher_pid)/'cmdline'
            try:cmd=proc.read_bytes().replace(b'\x00',b' ').decode()
            except FileNotFoundError:break
            if str(WORK/'train.py') not in cmd or '--smoke' not in cmd:break
            time.sleep(10)
        result=json.loads((validation/'status.json').read_text())
        if result['phase']!='smoke_passed':raise RuntimeError(f'Validation did not pass: {result}')
        if json.loads((WORK/'config.json').read_text())!=cfg:raise RuntimeError('Configuration changed while validating')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(cfg['physical_gpus'][0]),PYTHONPATH=str(WORK)+os.pathsep+str(ROOT),
            PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        state('verifying_export')
        with (WORK/'logs/export_2gpu07_step1000_20261019.log').open('w') as stream:
            subprocess.run([cfg['model_python'],'-B','-m','a0_single_latent.verify_export','--run',str(validation)],env=env,cwd=WORK,stdout=stream,stderr=subprocess.STDOUT,check=True)
        if not json.loads((validation/'export_verification.json').read_text())['passed']:raise RuntimeError('Export parity failed')
        state('formal_training_and_evaluation',log=str(WORK/'logs/train_step1000_2gpu07_20261019.log'))
        with (WORK/'logs/train_step1000_2gpu07_20261019.log').open('a') as stream:
            subprocess.run(['bash',str(WORK/'run_1000steps.sh')],env=env,cwd=WORK,stdout=stream,stderr=subprocess.STDOUT,check=True)
        state('completed')
    except BaseException as error:
        state('failed',error=repr(error));raise

if __name__=='__main__':main()
