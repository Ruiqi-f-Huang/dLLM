import math
import unittest
import torch
from grpo_trajectory import transition_masks,kernel_log_probability


class TrajectoryTests(unittest.TestCase):
    def test_absorbing_transition_partition_and_kernel(self):
        before=torch.tensor([1,9,9,4]);after=torch.tensor([1,2,9,4])
        masked,new,remain=transition_masks(before,after,1,9)
        self.assertEqual(masked.tolist(),[False,True,True,False])
        logps=torch.tensor([-1.,math.log(.3),-2.,-3.],dtype=torch.float64)
        value=kernel_log_probability(logps,new,remain,.25)
        self.assertAlmostEqual(value,math.log(.25*.3)+math.log(.75))

    def test_revealed_tokens_and_prefix_cannot_change(self):
        with self.assertRaises(ValueError):
            transition_masks(torch.tensor([1,3,9]),torch.tensor([1,4,9]),1,9)
        with self.assertRaises(ValueError):
            transition_masks(torch.tensor([1,9]),torch.tensor([2,3]),1,9)

    def test_final_transition_and_empty_state(self):
        x=torch.tensor([-1.,-2.]);new=torch.tensor([True,False]);stay=torch.tensor([False,False])
        self.assertEqual(kernel_log_probability(x,new,stay,1.),-1.)
        with self.assertRaises(ValueError):
            kernel_log_probability(x,new,torch.tensor([False,True]),1.)
        empty=torch.zeros(2,dtype=torch.bool)
        self.assertEqual(kernel_log_probability(x,empty,empty,1.),0.)


if __name__=='__main__':unittest.main()
