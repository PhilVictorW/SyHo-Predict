"""Read-only terminal monitor; it never starts or restarts a job."""
import argparse
import json
import time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--run-dir",required=True)
    p.add_argument("--interval",type=float,default=5);p.add_argument("--once",action="store_true")
    a=p.parse_args();root=Path(a.run_dir)
    while True:
        print("\033[2J\033[H",end="")
        print("RoboTwin: Goal->Local preparation -> Stage III")
        for path in (root/"workflow_status.json",root/"interaction/status.json",root/"rl/status.json",root/"rl/last_validation.json"):
            if path.exists():
                print("\n"+str(path.relative_to(root)))
                print(json.dumps(json.loads(path.read_text()),ensure_ascii=False,indent=2))
        for phase in ("interaction","rl"):
            path=root/phase/"metrics.jsonl"
            if path.exists():
                print("\n"+phase+" recent metrics")
                for line in path.read_text().splitlines()[-5:]:print(line)
        if a.once:break
        time.sleep(max(1,a.interval))
if __name__=="__main__":main()
