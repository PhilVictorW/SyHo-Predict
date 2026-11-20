import torch
from torch import nn
from torch.nn import functional as F
from robotwin.data import DELTAS
from robotwin.round3.model import TerminalHoldDataset,TerminalHoldTeacher

class FutureActionDataset(TerminalHoldDataset):
    def sample(self,e,t):
        b=super().sample(e,t)
        indices=t+DELTAS;valid=indices<e['length']
        target=torch.zeros(6,16)
        if valid.any():
            raw=torch.from_numpy(self.actions[e['dataset_from_index']+indices[valid]].copy())
            target[torch.as_tensor(valid)]=self.normalizer.normalize(raw)
        b['future_action_target']=target
        b['future_action_valid']=torch.as_tensor(valid)
        return b

class GroundedTeacher(TerminalHoldTeacher):
    def __init__(self,cfg,auxiliary=True):
        super().__init__(cfg)
        self.auxiliary=auxiliary
        self.aux_weight=0.
        # Both arms inherit exactly the same actor; head creation does not alter its RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4026)
            self.local_head=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256),nn.GELU(),nn.Linear(256,16))
            self.goal_head=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256),nn.GELU(),nn.Linear(256,80))
            with torch.no_grad():
                for head,n in [(self.local_head,1),(self.goal_head,5)]:
                    nn.init.normal_(head[-1].weight,std=.001);head[-1].bias.zero_()
                    bias=head[-1].bias.view(n,16);bias[:,[3,11]]=1.
        self.local_head.requires_grad_(auxiliary);self.goal_head.requires_grad_(auxiliary)

    def conditions_encoded(self,batch,prepared,donor=None,roles_off=False):
        p=self.policy
        source_batch,source_prepared=(batch,prepared) if donor is None else donor
        # Use recipient validity even for a fully valid wrong-future donor.
        g,l=p.future(*source_prepared['future'],batch['local_valid'].to(p.device),batch['goal_frame_valid'].to(p.device))
        enc=p.language(prepared['current'],g,l)
        valid=torch.stack((batch['goal_frame_valid'].any(-1),batch['local_valid']),-1).to(p.device)
        if roles_off:valid=torch.zeros_like(valid)
        return (*prepared['base'],self.flow.goal_proj(enc['goal']),self.flow.local_proj(enc['local']),valid),enc

    def conditions(self,batch,prepared,donor=None,roles_off=False):
        return self.conditions_encoded(batch,prepared,donor,roles_off)[0]

    def auxiliary_loss(self,batch,encoded):
        prediction=torch.cat((self.local_head(encoded['local'].float().mean(1))[:,None],
                              self.goal_head(encoded['goal'].float().mean(1)).view(-1,5,16)),1).float()
        target=batch['future_action_target'].to(prediction.device)
        valid=batch['future_action_valid'].to(prediction.device)
        # Normalize each term by its own valid dimensions; invalid future frames never contribute.
        denom=valid.sum().clamp_min(1)
        def mse(ids):return (((prediction[...,ids]-target[...,ids]).square().mean(-1))*valid).sum()/denom
        position=mse([0,1,2,8,9,10]);gripper=mse([7,15])
        rotation=[]
        for j in [3,11]:
            q=F.normalize(prediction[...,j:j+4],dim=-1,eps=1e-6)
            t=F.normalize(target[...,j:j+4],dim=-1,eps=1e-6)
            rotation.append((1-(q*t).sum(-1).square().clamp(0,1)))
        rotation=(torch.stack(rotation,-1).mean(-1)*valid).sum()/denom
        return dict(aux=(position+gripper+rotation)/3,aux_position=position,aux_rotation=rotation,aux_gripper=gripper)

    def forward(self,batch):
        prep=self.prepare(batch);conditions,encoded=self.conditions_encoded(batch,prep)
        flow=self.flow.loss(batch['actions'].cuda(),batch['actions_mask'].cuda(),batch['action_hz'].cuda(),
            batch['embodiment_id'].cuda(),batch['remaining_actions'].cuda(),conditions,repeats=2)
        if self.auxiliary:aux=self.auxiliary_loss(batch,encoded)
        else:aux={k:flow.detach()*0 for k in ['aux','aux_position','aux_rotation','aux_gripper']}
        return dict(flow=flow,**aux)
