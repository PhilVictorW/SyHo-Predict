"""Student-only RoboTwin websocket server, compatible with the benchmark client."""
import argparse,json,logging,os,time
from pathlib import Path
import numpy as np
import torch
from .model import RobotwinPolicy
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

class StudentServerPolicy:
    def __init__(self,path):
        self.path=Path(path).resolve()
        self.policy=RobotwinPolicy.load(self.path.parent).to('cuda:0').eval().requires_grad_(False)
        if self.policy.mode!='student':raise ValueError('Only Stage II student checkpoints can be deployed')
        if self.policy.cfg['dataset_type']!='robotwin_eef':raise ValueError('Wrong embodiment')
        self.calls=0
    @torch.inference_mode()
    def predict_action(self,examples,**kwargs):
        if not examples:raise ValueError('Empty batch')
        def image(x):
            x=np.asarray(x)
            if x.dtype!=np.uint8 or x.shape!=(256,256,3):raise ValueError(f'Expected uint8 RGB 256x256, got {x.shape}/{x.dtype}')
            return torch.from_numpy(x.copy()).permute(2,0,1)
        main=[];wrist=[];instructions=[]
        for e in examples:
            if any(k not in {'primary_image','wrist_image','lang','embodiment_id','action_hz'} for k in e):
                raise ValueError('Student input must contain only current cameras, language and control metadata')
            if len(e['primary_image'])!=1 or len(e['wrist_image'])!=2:raise ValueError('Three current views required')
            if int(e['embodiment_id'])!=1 or float(e['action_hz'])!=30.:raise ValueError('Expected Agilex at 30 Hz')
            main.append(image(e['primary_image'][0]));wrist.append(torch.stack([image(x) for x in e['wrist_image']]))
            instructions.append(str(e['lang']))
        batch=dict(current_main=torch.stack(main),current_wrist=torch.stack(wrist),instruction=instructions)
        policy=self.policy;b=len(examples);device=policy.device
        with torch.autocast('cuda',dtype=torch.bfloat16):
            conditions=policy.action_conditions(batch,policy.encode(batch))
            action=policy.flow.sample(conditions,torch.full((b,),30.,device=device),torch.ones(b,device=device,dtype=torch.long),steps=10)
        if action.shape!=(b,8,16) or not torch.isfinite(action).all():raise ValueError('Invalid student action output')
        self.calls+=1
        if self.calls==1:print(json.dumps(dict(first_inference=True,step=policy.source_step,shape=list(action.shape),teacher_loaded=False,future_input=False)),flush=True)
        # Client performs clipping, one denormalization, gripper inversion and quaternion normalization.
        return dict(normalized_actions=action.float().cpu().numpy())

def main():
    p=argparse.ArgumentParser();p.add_argument('--ckpt_path',required=True);p.add_argument('--port',required=True,type=int)
    p.add_argument('--idle_timeout',type=int,default=-1);p.add_argument('--use_bf16',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(2026)
    torch.cuda.set_per_process_memory_fraction(float(os.getenv('ROBOTWIN_SERVER_MEMORY_FRACTION','.14')))
    begin=time.time();policy=StudentServerPolicy(a.ckpt_path)
    metadata=dict(ckpt_path=str(policy.path),server_type='robotwin_goal_local_student',supported_eval_envs=['robotwin'],
        stage='II',step=policy.policy.source_step,teacher_loaded=False,uses_future=False,uses_state=False,action_hz=30,horizon=8)
    print(json.dumps(dict(ready=True,load_seconds=time.time()-begin,physical_gpu=os.getenv('CUDA_VISIBLE_DEVICES'),**metadata)),flush=True)
    WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,idle_timeout=a.idle_timeout,metadata=metadata).serve_forever()

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO);main()
