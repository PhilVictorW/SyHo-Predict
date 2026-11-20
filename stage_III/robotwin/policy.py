"""Same Round10 base; trainable queries/gate with a direct delta multiplier."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from robotwin.round7.model import CurrentQueryStudent, OBSERVATION_KEYS
from goal_local.roles import role_attention_mask
from .gate_actor import goal_local_update, pool_features

class InteractionPolicy(CurrentQueryStudent):
    def __init__(self, teacher):
        super().__init__(teacher)
        self.register_buffer("baseline_goal", torch.zeros(8, 2048), persistent=False)
        self.register_buffer("baseline_local", torch.zeros(8, 2048), persistent=False)

    @property
    def teacher(self):
        return self.__dict__["_teacher"]

    @torch.no_grad()
    def remember_baseline_queries(self):
        self.baseline_goal.copy_(self.q_goal)
        self.baseline_local.copy_(self.q_local)

    def set_training(self, enabled, variant="query_gate"):
        self.requires_grad_(False)
        self.teacher.requires_grad_(False).eval()
        self.train(enabled)
        self.teacher.flow.train(enabled)
        self.teacher.flow.DiT.eval()  # Preserve centered residual: no dropout drift.
        self.teacher.flow.checkpoint_blocks = enabled
        self.checkpoint_blocks = enabled
        if enabled:
            self.q_goal.requires_grad_(True); self.q_local.requires_grad_(True)
            self.gate.requires_grad_(variant == "query_gate")
        return {n:p for n,p in self.named_parameters() if p.requires_grad}

    def language_roles(self, current, baseline=False, eta=None, actor=None, gate_seed=None, mean=False):
        hidden, valid, am, fm, positions, visual_mask, deepstack = current
        b, length, _ = hidden.shape
        qg, ql = (self.baseline_goal, self.baseline_local) if baseline else (self.q_goal, self.q_local)
        hidden = torch.cat((hidden, qg[None].expand(b,-1,-1).to(hidden),
                            ql[None].expand(b,-1,-1).to(hidden)), 1)
        start = positions.amax((0,2))+1
        appended = start[:,None]+torch.arange(16,device=hidden.device)[None]
        positions = torch.cat((positions,appended[None].expand(positions.shape[0],-1,-1)), -1)
        visual_mask = F.pad(visual_mask,(0,16),value=False)
        attn = role_attention_mask(valid,dtype=hidden.dtype)
        lm = self.qwen.language_model
        rotary = lm.rotary_emb(hidden,positions[-3:])
        cp = torch.arange(hidden.shape[1],device=hidden.device)
        roles, gate_record, deep_eta = {}, None, (0. if baseline else eta)
        for i, block in enumerate(lm.layers):
            # During supervised checkpoint recomputation eta is fixed and no Actor runs.
            layer_eta = deep_eta
            def run(h, i=i, block=block, layer_eta=layer_eta):
                h = block(h,attention_mask=attn,position_ids=positions[0],
                          cache_position=cp,position_embeddings=rotary,use_cache=False)
                if i < len(deepstack):
                    h = lm._deepstack_process(h,visual_mask,deepstack[i])
                if i >= 8:
                    updated, _ = goal_local_update(self.gate,h[:,length:length+8],
                                                   h[:,length+8:],i-8,layer_eta)
                    h = torch.cat((h[:,:length+8],updated),1)
                return h
            hidden = checkpoint(run,hidden,use_reentrant=False) if self.checkpoint_blocks and self.training else run(hidden)
            if i == 7:
                features = pool_features(hidden,valid,length)
                if actor is not None:
                    if self.training or baseline or eta is not None:
                        raise ValueError("Actor is only used in frozen Student rollout")
                    gate_record = actor.decide(features,gate_seed,deterministic=mean)
                    deep_eta = gate_record["eta"]
                else:
                    gate_record = dict(features=features)
            if i+1 in self.cfg["distill_layers"]:
                roles[i+1] = (hidden[:,length:length+8],hidden[:,length+8:])
        hidden = lm.norm(hidden)
        roles[16] = (hidden[:,length:length+8],hidden[:,length+8:])
        return dict(goal=roles[16][0],local=roles[16][1],layers=roles,gate=gate_record)

    def encode_conditions(self, observations, prepared=None, **kw):
        if set(observations) != set(OBSERVATION_KEYS):
            raise ValueError("Student accepts current images and instruction only")
        prepared = self.prepare(observations) if prepared is None else prepared
        encoded = self.language_roles(prepared["current"], **kw)
        b = encoded["goal"].shape[0]
        valid = torch.ones(b,2,dtype=torch.bool,device=encoded["goal"].device)
        conditions = (*prepared["base"], self.flow.goal_proj(encoded["goal"]),
                      self.flow.local_proj(encoded["local"]), valid)
        return conditions, encoded, prepared

    @torch.no_grad()
    def posterior(self, batch, prepared):
        future = self.teacher.policy.encode_future_images(batch)
        return self.teacher.conditions_encoded(batch,dict(**prepared,future=future))

    @torch.no_grad()
    def predict(self, observations, noise, **kw):
        cond, encoded, _ = self.encode_conditions(observations, **kw)
        return self.flow.sample(cond,noise), encoded["gate"]
