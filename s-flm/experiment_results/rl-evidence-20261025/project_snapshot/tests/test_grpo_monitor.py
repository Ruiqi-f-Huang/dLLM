import unittest
from types import SimpleNamespace
import torch
from grpo_monitor import ratio_metrics, parameter_step, gradient_parts, code_regions, ast_signature


class MonitorTests(unittest.TestCase):
    def test_ratios_are_split_by_advantage_with_shared_denominator(self):
        old=torch.zeros(2,2)
        current=torch.tensor([[.4,0.],[-.4,0.]])
        valid=torch.tensor([[True,False],[True,False]])
        result=ratio_metrics(current,old,current,torch.tensor([.5,-.5]),valid,.2,.04)
        self.assertEqual(result['positive']['above_clip'],1)
        self.assertEqual(result['negative']['below_clip'],1)
        self.assertEqual(result['zero']['tokens'],0)
        self.assertAlmostEqual(result['all']['policy_loss_contribution'],
            result['positive']['policy_loss_contribution']+result['negative']['policy_loss_contribution'])

    def test_parameter_update_alignment(self):
        p=torch.nn.Parameter(torch.tensor([2.,3.]))
        result=parameter_step([p],[torch.tensor([1.,3.])],.1,torch.tensor([-1.,0.]))
        self.assertEqual(result['update_norm'],1)
        self.assertEqual(result['update_cosine_negative_policy_gradient'],1)
        self.assertEqual(result['gradient_clip_scale'],1)

    def test_unicode_docstrings_and_comments_are_not_code(self):
        text='def f():\n    """小明 buys apples"""\n    x=2 # comment\n    return x\n'
        regions=code_regions(text)
        self.assertFalse(regions[text.index('小明')])
        self.assertFalse(regions[text.index('#')])
        self.assertTrue(regions[text.index('x=2')])
        self.assertIsNone(code_regions('def broken('))
        self.assertEqual(ast_signature('def f():\n    """one"""\n    return 1'),
                         ast_signature('def f():\n    """two"""\n    return 1'))

    def test_gradient_decomposition_equals_full_gradient(self):
        from diffu_grpo import clipped_loss, proxy_logps
        class Model(torch.nn.Module):
            mask_index=2
            def __init__(self):
                super().__init__();self.w=torch.nn.Parameter(torch.tensor([.1,-.2,.0]))
            def noise(self,t):return None,torch.zeros_like(t)
            def _sigma_from_alphat(self,a):return a
            def forward(self,xt,sigma):return self.w.log_softmax(0).expand(*xt.shape,3)
        m=Model();tokens=torch.tensor([[0,1],[1,0]])
        lengths=torch.tensor([1,1]);valid=torch.tensor([[False,True],[False,True]])
        old=torch.full((2,2),-1.);ref=torch.full((2,2),-1.2);a=torch.tensor([.5,-.5])
        args=SimpleNamespace(microbatch=1,prompt_mask=0.,beta=.04,epsilon=.2)
        info,pg=gradient_parts(m,list(m.parameters()),tokens,lengths,old,ref,a,valid,42,args)
        self.assertTrue(m.w.grad is None)
        loss,_=clipped_loss(proxy_logps(m,tokens,lengths,0.,42),old,ref,a,valid,.2,.04)
        loss.backward()
        self.assertAlmostEqual(info['combined_gradient_norm'],m.w.grad.norm().item(),places=6)
        self.assertTrue(torch.isfinite(pg).all())


if __name__=='__main__':unittest.main()
