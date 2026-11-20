"""Check A1 masked BF16 attention backward against an FP32 reference."""
import os,sys,json
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="4"
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa
from torch.utils.checkpoint import checkpoint
from a1_local.numerics import configure_attention_backends
from ablation.aba1.plus_common.roles import role_attention_mask
print(json.dumps(configure_attention_backends()))
torch.set_num_threads(2);torch.manual_seed(2026)
for length in (160,176,192):
    valid=torch.ones(1,length-16,device="cuda",dtype=torch.bool)
    valid[:,:3]=False
    mask=role_attention_mask(valid,0,16,dtype=torch.bfloat16)
    q,k,v=[torch.randn(1,16,length,128,device="cuda",dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        out=checkpoint(lambda a,b,c:sdpa(a,b,c,attn_mask=mask),q,k,v,use_reentrant=False)
        loss=out.float().square().mean()
    loss.backward()
    ref=[x.detach().float().requires_grad_() for x in (q,k,v)]
    expected=sdpa(*ref,attn_mask=mask.float());expected.square().mean().backward()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(),expected,rtol=.04,atol=.008)
    for actual,wanted in zip((q,k,v),ref):
        assert torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad.float(),wanted.grad,rtol=.07,atol=2e-7)
    print(json.dumps(dict(length=length,loss=float(loss),max_output_error=float((out.float()-expected).abs().max()),finite_gradients=True)))
print("Masked attention backward regression passed")
