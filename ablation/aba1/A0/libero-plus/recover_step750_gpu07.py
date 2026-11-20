"""Archive redundant recovery shards without losing data; resume the saved step750."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path

WORK=Path(__file__).resolve().parent
RUN=WORK/'runs/single_latent16_dino_step1000_2gpu07_20261019'
ARCHIVE=Path('/tmp/wangchaoyang_lawam_a0_plus_archive_20261019')
AUDIT=RUN/'recovery_20261019'
STATE=WORK/'workflow_step1000_gpu07.json'

def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    AUDIT.mkdir(exist_ok=True)
    for name,source in [('failed_training_status.json',RUN/'status.json'),('failed_workflow_status.json',STATE),('resume_source.json',RUN/'latest.json')]:
        dest=AUDIT/name
        if not dest.exists():shutil.copy2(source,dest)
    cfg=json.loads((WORK/'config.json').read_text())
    latest=json.loads((RUN/'latest.json').read_text())
    if latest['step']!=750:raise ValueError('Expected the verified step750 checkpoint')
    def state(phase,**kw):
        row=dict(phase=phase,pid=os.getpid(),time=time.time(),gpus=[0,7],stop_after_updates=1000,run=str(RUN),resume_step=750,**kw)
        atomic(STATE,row);print(json.dumps(row),flush=True)
    try:
        state('archiving_redundant_recovery_shards')
        sources=[
            WORK/'runs/single_latent16_dino_1000epochs_20261019_smoke_r2/checkpoints/epoch_0000_step_000000002',
            WORK/'runs/single_latent16_dino_step1000_2gpu07_20261019_smoke/checkpoints/epoch_0000_step_000000004',
            RUN/'checkpoints/epoch_0000_step_000000500']
        archive_log=[]
        for checkpoint in sources:
            marker=json.loads((checkpoint/'checkpoint_complete.json').read_text())
            for name,info in sorted(marker['files'].items()):
                if not name.startswith('rank_'):continue
                source=checkpoint/name;dest=ARCHIVE/source.relative_to(WORK)
                dest.parent.mkdir(parents=True,exist_ok=True)
                if source.is_symlink():
                    if source.resolve()!=dest or dest.stat().st_size!=info['bytes'] or digest(dest)!=info['sha256']:raise ValueError('Invalid existing archive link')
                else:
                    if source.stat().st_size!=info['bytes']:raise ValueError('Unexpected source size')
                    if shutil.disk_usage(ARCHIVE).free<info['bytes']+10*1024**3:raise OSError('Archive filesystem lacks space')
                    temp=dest.with_suffix('.pt.copying')
                    h=hashlib.sha256();copied=0;last=0
                    print('Archiving '+str(source),flush=True)
                    with source.open('rb') as incoming,temp.open('wb') as outgoing:
                        for chunk in iter(lambda:incoming.read(16*1024*1024),b''):
                            outgoing.write(chunk);h.update(chunk);copied+=len(chunk)
                            if copied-last>=2*1024**3:
                                print(f'{name}: {copied/1024**3:.1f} GiB copied',flush=True);last=copied
                        outgoing.flush();os.fsync(outgoing.fileno())
                    if copied!=info['bytes'] or h.hexdigest()!=info['sha256'] or digest(temp)!=info['sha256']:
                        raise ValueError('Archive checksum mismatch; source retained')
                    shutil.copystat(source,temp);temp.replace(dest)
                    link=source.with_suffix('.pt.ssd_link')
                    link.symlink_to(dest)
                    os.replace(link,source)
                archive_log.append(dict(source=str(source),archive=str(dest),bytes=info['bytes'],sha256=info['sha256'],verified=True,original_path_preserved_by_symlink=True))
                atomic(AUDIT/'archive_manifest.json',archive_log)
                print(f'Archived {name}; HDD free {shutil.disk_usage(WORK).free/1024**3:.1f} GiB',flush=True)
        free=shutil.disk_usage(WORK).free/1024**3
        if free<65:raise OSError(f'Need sufficient checkpoint headroom after archiving, free={free:.1f} GiB')
        atomic(AUDIT/'recovery_plan.json',dict(reason='Final save failed the 45 GiB free-space guard',saved_step=750,target_step=1000,recomputed_updates=250,training_code_changed=False,config_changed=False,hdd_free_gib_before_resume=free,archive_manifest=str(AUDIT/'archive_manifest.json'),original_failed_attempt_reached_step=1000))
        state('resuming_training_and_evaluation',hdd_free_gib=free,checkpoint=latest['path'],log=str(WORK/'logs/resume_step750_gpu07_20261019.log'))
        with (WORK/'logs/resume_step750_gpu07_20261019.log').open('a') as stream:
            subprocess.run(['bash',str(WORK/'train.sh'),'--execute','--resume',latest['path'],'--after-eval'],cwd=WORK,stdout=stream,stderr=subprocess.STDOUT,check=True)
        state('completed')
    except BaseException as error:
        state('failed',error=repr(error));raise

if __name__=='__main__':main()
