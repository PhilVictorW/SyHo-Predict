import json,torch
from pathlib import Path
from transformers import AutoConfig,AutoModel,Qwen3VLModel
from ablation.aba1.plus_common.model import GoalLocalPolicy
cfg=json.loads(Path("config.json").read_text())
qc=AutoConfig.from_pretrained(cfg["qwen_path"],local_files_only=True);qc.text_config.num_hidden_layers=16
qc._attn_implementation="sdpa";qc.text_config._attn_implementation="sdpa";qc.vision_config._attn_implementation="sdpa"
torch.distributions.Distribution.set_default_validate_args(False)
with torch.device("meta"):
 torch.set_default_dtype(torch.bfloat16);q=Qwen3VLModel(qc);torch.set_default_dtype(torch.float32)
 d=AutoModel.from_config(AutoConfig.from_pretrained(cfg["dino_path"],local_files_only=True))
 model=GoalLocalPolicy(cfg,q,d,None,json.loads(Path(cfg["statistics_path"]).read_text()),"posterior")
nb=lambda ps:sum(p.numel()*p.element_size() for p in ps)
p=nb(model.parameters());g=nb(x for x in model.parameters() if x.requires_grad)
f=nb(model.qwen.visual.parameters())+nb(model.dino.parameters());future=nb(model.future.parameters());flow=nb(model.flow.parameters())
s=p-future+16*2048*4;sg=g-future+16*2048*4;t=p-flow
out=dict(teacher_model_gib=p/2**30,teacher_grads_gib=g/2**30,student_model_gib=s/2**30,student_grads_gib=sg/2**30,frozen_encoder_gib=f/2**30,stageI_static_four_ranks_gib=4*(p+g)/2**30,stageII_static_four_ranks_gib=4*(s+sg+t)/2**30,stageII_shared_frozen_static_four_ranks_gib=4*(s+sg+t-f)/2**30)
Path("memory_estimate.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
