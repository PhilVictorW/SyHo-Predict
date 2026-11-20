import argparse,json,os,sys
from pathlib import Path
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK.parents[3]))
import torch
from a0_single_latent.model import A0Policy
from a0_single_latent.config import atomic_json
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
torch.set_num_threads(2)
latest=json.loads((run/'latest.json').read_text());model=A0Policy.load(latest['path']).to('cuda:0').eval().requires_grad_(False)
reference=torch.load(run/'inference_reference_rank_0.pt',map_location='cpu',weights_only=False)
actions=model.predict_action(reference['main'],reference['wrist'],reference['instruction'],reference['noise'].to('cuda:0')).cpu()
expected=reference['actions']
error=(actions-expected).abs()
torch.testing.assert_close(actions,expected,rtol=.03,atol=.03)
report=dict(checkpoint=latest['path'],shape=list(actions.shape),max_absolute_error=error.max().item(),
            no_dino=not any('dino' in k for k in model.state_dict()),latent_shape=list(model.latent_query.shape),
            success=True,comparison='independent single-GPU deployment export vs restored three-rank model, fixed zero noise')
atomic_json(run/'export_verification.json',report);print(json.dumps(report,indent=2))
