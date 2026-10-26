import unittest
import torch
from grpo_controlled import signed_loss, gradient_geometry


class ControlledTests(unittest.TestCase):
    def test_program_gradient_rewards_success_and_penalizes_failure(self):
        x=torch.tensor([[-2.,-3.],[-4.,-5.]],requires_grad=True)
        valid=torch.tensor([[True,True],[True,False]])
        loss=signed_loss(x,valid,torch.tensor([.5,-.5]),'program')
        grad=torch.autograd.grad(loss,x)[0]
        self.assertLess(grad[0,0],0)
        self.assertGreater(grad[1,0],0)
        self.assertEqual(grad[1,1],0)
        self.assertAlmostEqual(loss.item(),-.5*((-2.5)-(-4.))/2)

    def test_ignored_negative_infinity_and_microbatch_equivalence(self):
        x=torch.tensor([[-2.,-3.],[-4.,-float('inf')]],requires_grad=True)
        valid=torch.tensor([[True,True],[True,False]])
        a=torch.tensor([.5,-.5])
        for mode in ['program','token']:
            whole=signed_loss(x,valid,a,mode)
            parts=0
            for i in range(2):
                weight=.5 if mode=='program' else valid[i].sum()/valid.sum()
                parts=parts+signed_loss(x[i:i+1],valid[i:i+1],a[i:i+1],mode)*weight
            torch.testing.assert_close(whole,parts)
            self.assertTrue(torch.isfinite(torch.autograd.grad(whole,x,retain_graph=True)[0]).all())

    def test_gradient_geometry_handles_zero_reference(self):
        result=gradient_geometry(torch.tensor([1.,2.]),torch.zeros(2))
        self.assertEqual(result['reference_gradient_norm'],0)
        self.assertIsNone(result['cosine'])
        result=gradient_geometry(torch.tensor([1.,0.]),torch.tensor([-2.,0.]))
        self.assertEqual(result['cosine'],-1)
        self.assertEqual(result['reference_to_policy_norm'],2)


if __name__=='__main__':unittest.main()
