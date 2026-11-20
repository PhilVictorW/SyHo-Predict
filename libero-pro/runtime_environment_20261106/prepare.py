"""Create immutable, shared Environment tasks and separate supplemental queues."""
import hashlib, sqlite3
from common import *
ROOT=BASE/'resources/environment_20261106'
ORIGINAL_RUN=BASE/'runs/20261105_233840_fixed2'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (RUN/'manifest.json').exists(),'Already prepared'
    original=read(ORIGINAL_RUN/'manifest.json')
    rendered=read(ROOT/'render_validation.json',{})
    assert rendered.get('status')=='passed' and rendered['tasks']==40,'Renderer validation incomplete'
    tasks=[];jobs=[]
    reports=[]
    for si,suite in enumerate(SUITES):
        rows=read(ROOT/('validated_'+suite+'.json'))
        assert len(rows)==10
        for r in sorted(rows,key=lambda x:x['task_id']):
            assert r['status']=='validated' and r['initial_states']==50 and r['unique_states']==50
            assert r['source_environment']!=r['target_environment']
            for path_key,hash_key in [('source','source_sha256'),('source_init','source_init_sha256'),('bddl','bddl_sha256'),('init','init_sha256'),('fixtures','fixtures_sha256')]:
                assert digest(r[path_key])==r[hash_key]
            task={key:r[key] for key in ('suite','task_id','name','instruction','bddl','init','initial_states','state_shape','bddl_sha256','init_sha256','fixtures','fixtures_sha256','source_environment','target_environment')}
            task.update(pro_suite=suite+'_env',category='env')
            tasks.append(task);reports.append(r)
            for ei in range(50):
                jobs.append(dict(task,id=8000+len(jobs),episode_idx=ei,noise_seed=5_000_000_000+(si+1)*10_000_000+r['task_id']*10000+ei*100))
    assert len(tasks)==40 and len(jobs)==2000
    m={k:v for k,v in original.items() if k not in ('tasks','jobs','total_tasks','total_episodes','scope_note','dataset_revision')}
    m.update(tasks=tasks,jobs=jobs,total_tasks=40,total_episodes=2000,dataset_revision=None,
             source_type='locally_generated_environment_extension',
             scope_note='Environment extension generated from pinned official source. Not a released Hugging Face Environment split. Non-living-room tasks switch to living_room_table; five living-room tasks switch to main_table to avoid identity perturbation. The 50 released base states per task are translated according to workspace and support surfaces, validated, and supplemented only when necessary. Fixed fixture poses are explicitly restored. Both weights use identical 2000 states and noise seeds.',
             generation_reports=str(ROOT),original_four_category_run=str(ORIGINAL_RUN))
    protocol=dict(github_commit=original['github_commit'],generator_sha256=digest(Path(__file__).parent/'generate.py'),
        official_perturbation_sha256=digest(REPO/'perturbation.py'),official_environment_config_sha256=digest(REPO/'libero_ood/ood_environment.yaml'),
        target_rule={'living_room_table':'main_table','all_other_source_environments':'living_room_table'},
        source_type=m['source_type'],evaluation_config={k:m[k] for k in ('max_steps','settling_steps','image_size','action_chunk','flow_steps','uses_state','uses_future','teacher_loaded','stage_III_actor')},
        state_validation=reports[0]['validation'],reports=reports,render_validation=str(ROOT/'render_validation.json'),
        checkpoints=CHECKPOINTS,shared_states_and_policy_noise=True,fixed_fixture_poses_saved_and_restored=True,robot_initialization_noise='preserved_from_released_base_states; optional 0.003-rad jitter only if extra candidates needed',gpu_assignment=GPUS,max_inference_processes_per_gpu=2,
        original_data_preserved=True,combined_episodes_per_model=10000,created=time.time())
    RUN.mkdir(parents=True,exist_ok=True)
    for variant,gpus in GPUS.items():
        folder=RUN/variant;folder.mkdir(exist_ok=True)
        for sub in ('episodes','workers','errors','processes'): (folder/sub).mkdir(exist_ok=True)
        path=folder/'queue.sqlite';assert not path.exists()
        db=sqlite3.connect(path);db.execute('PRAGMA journal_mode=WAL')
        db.execute("CREATE TABLE jobs(id INTEGER PRIMARY KEY,payload TEXT NOT NULL,gpu INTEGER,state TEXT NOT NULL DEFAULT 'pending',worker TEXT,attempts INTEGER NOT NULL DEFAULT 0,success INTEGER,error TEXT,started REAL,finished REAL)")
        db.execute('CREATE INDEX job_queue ON jobs(state,gpu,id)')
        db.executemany('INSERT INTO jobs(id,payload,gpu) VALUES(?,?,?)',[(j['id'],json.dumps(j),gpus[i%len(gpus)]) for i,j in enumerate(jobs)])
        db.commit()
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]==2000
        db.close()
    atomic(RUN/'manifest.json',m);atomic(RUN/'protocol.json',protocol)
    atomic(RUN/'launch.json',dict(prepared=time.time(),gpu_assignment=GPUS,phase='prepared'))
    print(json.dumps(dict(run=str(RUN),tasks=40,episodes_per_model=2000,gpus=GPUS)))
if __name__=='__main__': main()
