import unittest
from types import SimpleNamespace
import torch
from diffu_grpo import partial_completion_mask, proxy_logps, completion_mask

class PartialTrainingTests(unittest.TestCase):
    def test_count_reuse_and_scope(self):
        valid=torch.tensor([[False]+[True]*8+[False]*3]*2)
        hidden=partial_completion_mask(valid,.75,42)
        self.assertEqual(hidden.sum(1).tolist(),[6,6])
        self.assertFalse(bool((hidden & ~valid).any()))
        self.assertTrue(torch.equal(hidden,partial_completion_mask(valid,.75,42)))
        self.assertTrue(torch.equal(valid,partial_completion_mask(valid,1.,42)))
    def test_proxy_context_and_gradient(self):
        class Model:
            mask_index=9
            tokenizer=SimpleNamespace(eos_token_id=2,pad_token_id=0)
            def noise(self,t):return t,t*0
            def _sigma_from_alphat(self,a):return a
            def forward(self,xt,sigma):
                self.xt=xt
                return torch.log_softmax(torch.zeros((*xt.shape,10),requires_grad=True),-1)
        model=Model();tokens=torch.tensor([[1,3,4,5,6,7,8,2,0,0]])
        lengths=torch.tensor([2]);valid=completion_mask(tokens,lengths,2,0,9)
        chosen=partial_completion_mask(valid,.75,13)
        scores=proxy_logps(model,tokens,lengths,0.,13,.75)
        self.assertTrue(torch.equal(model.xt[valid & ~chosen],tokens[valid & ~chosen]))
        self.assertTrue(bool(model.xt[chosen].eq(9).all()))
        self.assertEqual(model.xt[0,:2].tolist(),[1,3])
        self.assertEqual(model.xt[0,-2:].tolist(),[9,9])
        scores[chosen].sum().backward()

if __name__=='__main__':unittest.main()
