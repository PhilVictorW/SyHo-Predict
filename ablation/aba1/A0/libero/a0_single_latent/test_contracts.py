import collections,types,unittest
import torch
from torch import nn
from torch.nn import functional as F
from .config import WORK,load_config
from .distributed import ExactRankBatchSampler
from .attention import latent_attention_mask,remove_old_queries
from .model import A0Policy

class Block(nn.Module):
    def __init__(self):
        super().__init__();self.qkv=nn.Linear(32,96);self.out=nn.Linear(32,32)
    def forward(self,x,attention_mask=None,**kwargs):
        q,k,v=self.qkv(x).chunk(3,-1)
        return x+self.out(F.scaled_dot_product_attention(q[:,None],k[:,None],v[:,None],attn_mask=attention_mask)[:,0])
class Qwen(nn.Module):
    def __init__(self):
        super().__init__();self.visual=nn.Linear(32,32)
        self.language_model=nn.Module();self.language_model.layers=nn.ModuleList([Block() for _ in range(2)])
        self.language_model.norm=nn.LayerNorm(32);self.language_model.rotary_emb=lambda *a:None
        self.config=types.SimpleNamespace(text_config=types.SimpleNamespace(hidden_size=32))

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_exact_epoch_and_tail_weights(self):
        for size in (1,22,32,54,260726):
            samples=[];sums=collections.defaultdict(float);lengths=[]
            for rank in range(3):
                seq=list(ExactRankBatchSampler(size,rank,2026,3));lengths.append(len(seq))
                update=0
                for (index,w,n,last), in seq:
                    if w:samples.append(index)
                    sums[update]+=w/3
                    if last:update+=1
            self.assertEqual(len(set(lengths)),1)
            self.assertEqual(sorted(samples),list(range(size)))
            self.assertTrue(all(abs(v-1)<1e-7 for v in sums.values()))
        for rank in range(3):
            full=list(ExactRankBatchSampler(54,rank,1,0))
            self.assertEqual(list(ExactRankBatchSampler(54,rank,1,0,cursor=32)),full[11:])
    def test_single_group_mask_and_query_removal(self):
        valid=torch.ones(1,20,dtype=torch.bool);valid[:,1]=False
        mask=latent_attention_mask(valid)[0,0]
        self.assertTrue(torch.isneginf(mask[:20,20:]).all())
        self.assertTrue(torch.isfinite(mask[20:,20:]).all())
        self.assertTrue(torch.isneginf(mask[20:,1]).all())
        am=torch.zeros_like(valid);fm=am.clone();am[:,2:10]=True;fm[:,10:18]=True
        x=torch.arange(20).reshape(1,20,1)
        out,_=remove_old_queries(x,valid,am,fm)
        self.assertEqual(out.flatten().tolist(),[0,1,18,19])
    def test_action_only_gradient_and_current_only_inference(self):
        cfg=load_config(WORK/'config.json')
        cfg['flow']=dict(cfg['flow'],vlm_dim=32,vision_dim=32,hidden_dim=32,attention_heads=4,num_layers=4,num_vision_tokens=2)
        stats={'franka':{'action':{'min':[-1.]*7,'max':[1.]*7}}}
        model=A0Policy(cfg,Qwen(),None,stats)
        model.encode_current_inputs=lambda *a:None
        def current(_):
            hidden=torch.cat((torch.ones(1,4,32),model.act_query[None],model.flow_action_query[None]),1)
            valid=torch.ones(1,20,dtype=torch.bool);am=torch.zeros_like(valid);fm=am.clone()
            am[:,4:12]=True;fm[:,12:]=True
            return hidden,valid,am,fm,torch.arange(20)[None,None].expand(3,1,-1),torch.zeros_like(valid),[]
        model.current_embeddings=current
        batch=dict(current_main=torch.zeros(1,3,256,256,dtype=torch.uint8),current_wrist=torch.zeros(1,3,256,256,dtype=torch.uint8),instruction=['move'],
            actions=torch.randn(1,50,32),actions_mask=torch.zeros(1,50,32,dtype=torch.bool),action_hz=torch.tensor([20.]),
            embodiment_id=torch.tensor([25]),remaining_actions=torch.tensor([3]))
        batch['actions_mask'][:,:3,:7]=True
        result=model(batch);self.assertEqual(set(result),{'loss','flow'});result['loss'].backward()
        for p in (model.latent_query,model.act_query,model.flow_action_query):
            self.assertGreater(p.grad.abs().sum().item(),0)
        self.assertFalse(any(k.startswith(('dino.','gate.','future.')) for k in model.state_dict()))
        self.assertFalse(hasattr(model.flow,'plan'));self.assertFalse(hasattr(model.flow,'cfg_embeddings'))
        a=model.predict_action(batch['current_main'],batch['current_wrist'],batch['instruction'],torch.zeros(1,50,32))
        self.assertEqual(a.shape,(1,8,7));self.assertTrue(torch.isfinite(a).all())
if __name__=='__main__':unittest.main()
