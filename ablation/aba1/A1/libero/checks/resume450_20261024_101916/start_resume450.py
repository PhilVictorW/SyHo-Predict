from pathlib import Path
import json,os,shutil,signal,subprocess,time,shlex,sys
W=Path('/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/ablation/aba1/A1/libero')
sys.path[:0]=[str(W),str(W.parents[3])]
from a1_local.config import atomic_json
python='/home/wangchaoyang/envs/LaWAM/bin/python'
previous=json.loads((W/'nohup/current.json').read_text());old=Path(previous['run'])
assert old.name=='local16_teacher1000_student2000_gpu012_stageII_save50_keep1_nohup_20261023_204059'
assert json.loads((old/'pipeline.json').read_text())['phase']=='failed'
assert not Path('/proc',str(previous['processes']['training']['pid'])).exists()
source=old/'stage_II/checkpoints/step_000450'
assert (source/'model.pt').stat().st_size==4892916418
backup=Path('/tmp/a1_resume450_backup_path').read_text()
stamp=time.strftime('%Y%m%d_%H%M%S')
name='local16_teacher1000_student2000_gpu012_resume450_nohup_'+stamp
run=W/'runs'/name;control=W/'nohup'/name
run.mkdir();control.mkdir()
def copy_file(src,dst):
    if Path(src).name=='model.pt': os.link(src,dst);return dst
    return shutil.copy2(src,dst)
shutil.copytree(old/'stage_I',run/'stage_I',copy_function=copy_file)
for f in ('latest.json','final.json'):
    item=json.loads((run/'stage_I'/f).read_text())
    item['path']=str(run/'stage_I/checkpoints/step_001000')
    atomic_json(run/'stage_I'/f,item)
atomic_json(run/'stage_I/imported_from.json',dict(run=str(old),checkpoint=str(old/'stage_I/checkpoints/step_001000'),time=time.time()))
plan=json.loads((W/'run_plan.json').read_text());plan['run_name']=name;plan.pop('resume_student',None)
atomic_json(run/'run_plan.json',plan);atomic_json(W/'run_plan.json',plan)
atomic_json(Path(backup)/'validation_summary.json',dict(tests_passed=17,resume_step=450,next_step=451,optimizer_restored=False,sample_offset_restored=True,learning_rate_offset_restored=True,checkpoint_every=50,checkpoint_keep=1))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
commands={
    'training':['nohup',python,'-B','-u','launch.py','--execute','--start-stage','II','--resume-student',str(source)],
    'progress':['nohup',python,'-B','-u','monitor.py','--run',str(run),'--interval','30'],
    'auto_eval':['nohup',python,'-B','-u','watch_full_eval.py','--execute'],
}
processes={}
for key,command in commands.items():
    log=control/(key+'.log')
    with log.open('ab',buffering=0) as stream:
        child=subprocess.Popen(command,cwd=W,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    processes[key]=dict(pid=child.pid,command=command,log=str(log))
record=dict(run=str(run),run_name=name,started_at=time.time(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    physical_gpus=[0,1,2],training_ranks=3,start_stage='II',resume_step=450,resume_source=str(source),
    optimizer_restored=False,model_load_strategy='rank_serial',load_gap_seconds=10,checkpoint_every=50,checkpoint_keep=1,
    processes=processes)
atomic_json(control/'launch_record.json',record);atomic_json(W/'nohup/current.json',record)
atomic_json(Path(backup)/'restart_record.json',record)
# Only stop the two obsolete readers tied to the failed A1 run.
for key,expected in [('progress',b'monitor.py'),('auto_eval',b'watch_full_eval.py')]:
    pid=previous['processes'][key]['pid'];proc=Path('/proc')/str(pid)
    if proc.exists():
        cmd=proc.joinpath('cmdline').read_bytes()
        if expected not in cmd or proc.joinpath('cwd').resolve()!=W:raise RuntimeError('Reader identity mismatch')
        os.kill(pid,signal.SIGTERM)
session='a1_libero_gpu012'
windows=[
    [python,'-B','-u',str(W/'tmux_progress.py'),'--run',str(run),'--interval','5'],
    ['tail','-n','60','-F',str(control/'training.log')],
    ['tail','-n','60','-F',str(control/'auto_eval.log')],
]
existing=subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0
for index,command in enumerate(windows):
    if existing:
        old_command=subprocess.check_output(['tmux','display-message','-p','-t',f'{session}:{index}','#{pane_start_command}'],text=True)
        if str(W) not in old_command:raise RuntimeError('Foreign tmux window; do not replace')
        args=['tmux','respawn-window','-k','-t',f'{session}:{index}','-c',str(W),shlex.join(command)]
    elif index==0:
        args=['tmux','new-session','-d','-s',session,'-n','progress','-c',str(W),shlex.join(command)]
    else:
        args=['tmux','new-window','-t',f'{session}:{index}','-n',('train-log' if index==1 else 'eval-log'),'-c',str(W),shlex.join(command)]
    subprocess.run(args,check=True)
subprocess.run(['tmux','select-window','-t',f'{session}:0'],check=True)
print(json.dumps(record,indent=2))
