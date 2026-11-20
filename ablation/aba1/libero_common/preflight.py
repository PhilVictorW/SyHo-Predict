"""Read-only readiness checks. Does not load weights or initialize CUDA/simulation."""
import argparse
import importlib.util
import json
from pathlib import Path
from .config import ROOT, load_config
from .training import validate_paths,epoch_updates


def main(variant,config,argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json-output',help='Optional readiness report file, never a training run')
    args=parser.parse_args(argv)
    cfg=load_config(config,variant);validate_paths(cfg)
    dependencies={name:importlib.util.find_spec(name) is not None for name in
                  ['torch','transformers','numpy','pyarrow','av','PIL','diffusers']}
    if not all(dependencies.values()): raise RuntimeError(dependencies)
    info=json.loads((Path(cfg['dataset_root'])/'meta/info.json').read_text())
    if info['fps']!=20: raise ValueError('Dataset fps mismatch')
    base=ROOT.parent/'LaWAM_aba2'
    resources={name:str(path) for name,path in {
        'simulator_python':Path('/home/wangchaoyang/envs/LIBERO4LaWAM/bin/python'),
        'libero_home':base/'runtime/LIBERO','libero_config':base/'runtime/libero_config'}.items()}
    if not all(Path(value).exists() for value in resources.values()): raise FileNotFoundError(resources)
    report=dict(variant=variant,config=str(config),local_slots=cfg['local_slots'],goal_slots=cfg['goal_slots'],
        interaction=cfg['interaction'],dependencies=dependencies,resources=resources,
        dataset_frames=info.get('total_frames'),dataset_episodes=info.get('total_episodes'),
        a0_default_epochs=500 if variant=='A0' else None,
        default_updates_per_epoch=epoch_updates(260726,2,16),
        full_weight_loading_tested=False,cuda_initialized=False,simulation_started=False,
        formal_training_started=False,formal_inference_started=False,
        remaining_launch_inputs=['output directory','GPU selection'] if variant=='A0' else
            ['stage I/II epoch budget','stage-II teacher checkpoint','output directory','GPU selection'])
    if args.json_output:
        from .config import atomic_json
        atomic_json(args.json_output,report)
    print(json.dumps(report,indent=2))
