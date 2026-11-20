"""Student-only action-chunk prediction from current main/wrist RGB images."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .model import GoalLocalPolicy

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--main',required=True);p.add_argument('--wrist',required=True);p.add_argument('--instruction',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(4)
    model=GoalLocalPolicy.load(args.checkpoint).to(args.device).eval()
    def image(path):return torch.from_numpy(np.asarray(Image.open(path).convert('RGB')).copy()).permute(2,0,1)[None]
    result=model.predict_action(image(args.main),image(args.wrist),[args.instruction])
    Path(args.output).write_text(json.dumps(dict(actions=result[0].cpu().tolist(),action_hz=20,shape=[8,7]),indent=2))
if __name__=='__main__':main()
