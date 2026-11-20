import json,time
from pathlib import Path
R=Path(__file__).resolve().parent
while True:
    try:s=json.loads((R/"tail_workflow/status.json").read_text())
    except:s={}
    print("\033[2J\033[H",end="")
    print("RoboTwin Round11 尾部采样对照训练 | GPU2 | "+time.strftime("%F %T"))
    print("workflow:",s.get("phase","starting"),"mode:",s.get("mode","-"),"step:",s.get("step",0),"/200")
    for mode in ["uniform","tail"]:
        try:d=json.loads((R/f"tail_adapter_200_{mode}/status.json").read_text());print(mode,d.get("phase"),d.get("step",0),"/200",d.get("validation_sample",""))
        except:print(mode,"waiting")
    print("["+("#"*min(40,int(40*s.get("step",0)/200))).ljust(40,".")+f"] {s.get('step',0)}/200")
    print("init: action_adapter_200/step_000200 | LR=5e-6 | batch=4 | save/validate=50")
    print("after both modes: three-way held-out confirmation; old StageII not restarted",flush=True)
    time.sleep(5)
