"""Resume A0 from step 750 after the user-authorized removal of step 500."""
import json
import os
from pathlib import Path
import subprocess
import time

WORK = Path(__file__).resolve().parent
RUN = WORK / 'runs/single_latent16_dino_step1000_2gpu07_20261019'
CHECKPOINT = RUN / 'checkpoints/epoch_0000_step_000000750'
STATE = WORK / 'workflow_step1000_gpu07.json'
LOG = WORK / 'logs/resume_after_delete_step500_gpu07_20261019.log'


def state(phase, **extra):
    row = dict(phase=phase, pid=os.getpid(), time=time.time(), gpus=[0, 7],
               run=str(RUN), resume_step=750, stop_after_updates=1000,
               log=str(LOG), **extra)
    temporary = STATE.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(row, indent=2) + '\n')
    temporary.replace(STATE)
    print(json.dumps(row), flush=True)


def main():
    cfg = json.loads((WORK / 'config.json').read_text())
    assert cfg['physical_gpus'] == [0, 7] and cfg['stop_after_updates'] == 1000
    assert not (RUN / 'checkpoints/epoch_0000_step_000000500').exists()
    assert json.loads((CHECKPOINT / 'checkpoint_complete.json').read_text())['step'] == 750
    try:
        state('resuming_training', checkpoint=str(CHECKPOINT))
        with LOG.open('a') as stream:
            subprocess.run(['bash', str(WORK / 'train.sh'), '--execute', '--resume',
                            str(CHECKPOINT), '--no-after-eval'], cwd=WORK,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
            final = json.loads((RUN / 'final.json').read_text())
            marker = json.loads((Path(final['path']) / 'checkpoint_complete.json').read_text())
            assert final['step'] == marker['step'] == 1000 and marker['complete']
            state('evaluating_step1000', final_checkpoint=final['path'])
            subprocess.run([cfg['model_python'], '-B', str(WORK / 'evaluate.py'),
                            '--config', str(WORK / 'config.json'), '--checkpoint',
                            final['path'], '--scope', 'full', '--gpus', '0,7', '--execute'],
                           cwd=WORK, stdout=stream, stderr=subprocess.STDOUT, check=True)
        state('completed', final_checkpoint=final['path'])
    except BaseException as error:
        state('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
