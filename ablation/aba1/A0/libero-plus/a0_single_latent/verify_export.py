import argparse,json
from pathlib import Path
import torch
from .model import A0Policy
from .deployment import validate_checkpoint
from .config import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    checkpoint=Path(json.loads((run/'latest.json').read_text())['path']);validate_checkpoint(checkpoint,verify_hash=True)
    ref=torch.load(run/'inference_reference_rank_0.pt',map_location='cpu',weights_only=False)
    torch.set_num_threads(2);model=A0Policy.load(checkpoint).to('cuda:0').eval()
    pred=model.predict_action(ref['main'],ref['wrist'],ref['instruction'],ref['noise'].to('cuda:0')).cpu()
    torch.testing.assert_close(pred,ref['actions'],atol=.01,rtol=.01)
    atomic_json(run/'export_verification.json',dict(passed=True,max_abs_difference=float((pred-ref['actions']).abs().max()),shape=list(pred.shape),uses_dino=True))
    print('checkpoint inference parity passed',flush=True)

if __name__=='__main__':main()
