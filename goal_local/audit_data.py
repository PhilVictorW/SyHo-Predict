import argparse,json
from pathlib import Path
from .data import FutureDataset

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/goal_local/default.json');p.add_argument('--output',default='artifacts/goal_local/data_audit.json');p.add_argument('--decode',action='store_true');args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());stats=json.loads(Path(cfg['statistics_path']).read_text())
    ds=FutureDataset(cfg['dataset_root'],stats,'all',decode=args.decode)
    report=ds.audit();examples=[]
    for e in (ds.episodes[0],ds.episodes[1],min(ds.episodes,key=lambda e:e['length']),ds.episodes[-1]):
        for t in sorted({0,max(0,e['length']-89),e['length']-9,e['length']-8,e['length']-1}):
            sample=ds.sample(e,t)
            examples.append(dict(episode=e['episode_index'],t=t,local_valid=bool(sample['local_valid']),goal_valid=sample['goal_frame_valid'].tolist(),valid_actions=int(sample['actions_mask'].any(-1).sum()),decoded=args.decode))
    report['edge_samples']=examples
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['validation_episodes','edge_samples']},indent=2))
if __name__=='__main__':main()
