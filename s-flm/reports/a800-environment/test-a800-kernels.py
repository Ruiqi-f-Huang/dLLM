import json,torch,flash_attn
from torch.nn import functional as F
assert torch.cuda.is_bf16_supported()
torch.manual_seed(42)
qkv=torch.randn(2,64,3,4,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
y=flash_attn.flash_attn_qkvpacked_func(qkv,dropout_p=0.,causal=False)
q,k,v=qkv.unbind(dim=2)
ref=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
torch.testing.assert_close(y,ref,rtol=.03,atol=.03)
y.float().square().mean().backward()
assert torch.isfinite(qkv.grad).all()
result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'flash_attn':flash_attn.__version__,'dtype':str(y.dtype),'max_abs_error':float((y-ref).abs().max()),'finite_gradients':True}
print(json.dumps(result,indent=2))
open('/root/setup-logs/a800-kernel-test.json','w').write(json.dumps(result,indent=2))
