"""Current-observation-only query student, sharing a frozen compatible actor."""
import copy
import torch
from torch import nn
from goal_local.model import GoalLocalPolicy
from goal_local.roles import RelevanceGate

OBSERVATION_KEYS=('current_main','current_wrist','instruction')

@torch.no_grad()
def prepare_current(teacher, observations):
    if set(observations)!=set(OBSERVATION_KEYS):
        raise ValueError('Current-only API accepts images and instruction only')
    p=teacher.policy
    inputs=p.encode_current_inputs(observations['current_main'],observations['current_wrist'],observations['instruction'])
    current=p.current_embeddings(inputs)
    hidden=p.qwen.get_input_embeddings()(inputs['input_ids']).clone()
    hidden[current[2]]=p.act_query[None].expand(len(hidden),-1,-1).reshape(-1,2048).to(hidden.dtype)
    hidden[current[3]]=p.flow_action_query[None].expand(len(hidden),-1,-1).reshape(-1,2048).to(hidden.dtype)
    p.qwen.rope_deltas=None
    original=p.qwen(inputs_embeds=hidden,attention_mask=inputs['attention_mask'],pixel_values=inputs['pixel_values'],
                    image_grid_thw=inputs['image_grid_thw'],use_cache=False).last_hidden_state
    dino=p.current_dino(observations['current_main'])
    predicted_future=teacher.decoder(dino,teacher.mapper(original[current[2]].reshape(len(hidden),8,2048)))[:,0]
    visual=torch.cat((dino,predicted_future),1)
    base=torch.cat((visual,teacher.flow.enc_vlm(original)),1)
    image_mask=torch.cat((torch.ones(visual.shape[:2],device=base.device,dtype=torch.bool),torch.zeros_like(current[1])),1)
    text_mask=torch.cat((torch.zeros(visual.shape[:2],device=base.device,dtype=torch.bool),current[1]),1)
    return dict(current=current,base=(base,image_mask,text_mask))

class CurrentQueryStudent(nn.Module):
    def __init__(self, frozen_teacher):
        super().__init__()
        assert not any(p.requires_grad for p in frozen_teacher.parameters())
        # Only student queries and its new relevance gate belong to this module.
        # The source actor is external, immutable, and never saved twice.
        self.__dict__['_teacher']=frozen_teacher
        self.cfg=copy.deepcopy(frozen_teacher.policy.cfg)
        self.mode='student';self.checkpoint_blocks=False
        self.q_goal=nn.Parameter(torch.zeros(8,2048))
        self.q_local=nn.Parameter(torch.zeros(8,2048))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(280901)
            self.gate=RelevanceGate(dim=2048,deep_layers=8,scale=0.)

    @property
    def qwen(self):return self.__dict__['_teacher'].policy.qwen
    @property
    def flow(self):return self.__dict__['_teacher'].flow

    def set_queries(self, goal, local):
        assert goal.shape==local.shape==(1,8,2048)
        with torch.no_grad():self.q_goal.copy_(goal[0]);self.q_local.copy_(local[0])

    def prepare(self,observations):return prepare_current(self.__dict__['_teacher'],observations)

    def conditions(self,prepared):
        assert set(prepared)=={'current','base'}
        b=prepared['current'][0].shape[0]
        encoded=GoalLocalPolicy.language(self,prepared['current'],self.q_goal[None].expand(b,-1,-1),self.q_local[None].expand(b,-1,-1))
        role_valid=torch.ones(b,2,device=encoded['goal'].device,dtype=torch.bool)
        return (*prepared['base'],self.flow.goal_proj(encoded['goal']),self.flow.local_proj(encoded['local']),role_valid)

    @torch.no_grad()
    def forward(self,observations,noise):
        return self.flow.sample(self.conditions(self.prepare(observations)),noise)
