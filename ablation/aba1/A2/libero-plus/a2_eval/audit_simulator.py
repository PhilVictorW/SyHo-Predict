"""Import simulator dependencies and read task/state metadata; never build an environment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from ablation.aba1.plus_common.config import ROOT,ABA,SUITES,MAX_STEPS,SUITE_COUNTS,load_config,atomic_json
from .runtime import subprocess_environment

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(Path(__file__).resolve().parents[1]/"config.json"))
    parser.add_argument("--worker",action="store_true");args=parser.parse_args()
    cfg=load_config(args.config);output=Path(__file__).resolve().parents[1]/"logs/full_eval_simulator_audit.json"
    if not args.worker:
        env=subprocess_environment(cfg,"",output.parent/"simulator_cache")
        subprocess.run([cfg["simulator_python"],"-B","-m","a2_eval.audit_simulator",
                        "--config",args.config,"--worker"],cwd=Path(__file__).resolve().parents[1],env=env,check=True)
        return
    from sim_compat import enable
    enable()
    import libero.libero as libero
    if not Path(libero.__file__).resolve().is_relative_to(Path(cfg["benchmark"])):
        raise ValueError("Wrong benchmark imported")
    from ablation.aba1.plus_common.environment import load_legacy_eval_core
    core=load_legacy_eval_core()
    import numpy as np
    import hashlib
    adapter=core.get_benchmark_adapter("libero_plus")
    limits={s:adapter.get_max_steps(s) for s in SUITES}
    if limits!=MAX_STEPS:raise ValueError("Environment horizons changed")
    benchmark=core._get_benchmark_dict();rows=[]
    for name in SUITES:
        import contextlib,io
        with contextlib.redirect_stdout(io.StringIO()):suite=benchmark[name]()
        if suite.n_tasks!=SUITE_COUNTS[name]:raise ValueError((name,suite.n_tasks))
        task=suite.get_task(0);states=suite.get_task_init_states(0)
        index=core._task_init_state_index(states,0)
        if index!=0:raise ValueError("Initial state index changed")
        a=np.ascontiguousarray(states[index])
        rows.append(dict(suite=name,tasks=suite.n_tasks,task0_name=task.name,init_shape=list(a.shape),
            state_sha256=hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()))
    report=dict(status="passed",simulator_python=sys.executable,libero_file=libero.__file__,eval_core_file=core.__file__,
        max_policy_steps=limits,suites=rows,sim_sysroot=cfg["sim_sysroot"],
        libero_config_path=cfg["libero_config_path"],environment_created=False,rollout_started=False)
    atomic_json(output,report);print(json.dumps(report,indent=2))

if __name__=="__main__":main()
