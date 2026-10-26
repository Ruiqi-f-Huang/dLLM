import unittest

import torch

from diffu_grpo import (clipped_loss, completion_mask, grouped_advantages,
                       perturb, reference_adapter)


class GRPOTests(unittest.TestCase):
    def test_eos_only_in_suffix_and_pad_excluded(self):
        tokens = torch.tensor([[2, 4, 5, 2, 6, 0], [4, 5, 6, 7, 0, 9]])
        valid = completion_mask(tokens, torch.tensor([2, 1]), 2, 0, 9)
        self.assertEqual(valid.tolist(), [[False, False, True, True, False, False],
                                         [False, True, True, True, False, False]])

    def test_advantage_is_centered_without_std_normalization(self):
        rewards = torch.tensor([1., 0., 0., 1., 1., 1.])
        advantage = grouped_advantages(rewards, 3)
        torch.testing.assert_close(advantage[:3], torch.tensor([2/3, -1/3, -1/3]))
        self.assertTrue(torch.equal(advantage[3:], torch.zeros(3)))

    def test_masks_are_reproducible_without_changing_global_rng(self):
        ids = torch.arange(24).reshape(2, 12)
        prefix = torch.tensor([3, 7])
        state = torch.random.get_rng_state()
        a = perturb(ids, prefix, 99, .5, 12)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(a, perturb(ids, prefix, 99, .5, 12)))
        for i, length in enumerate(prefix):
            self.assertTrue(torch.all(a[i, length:] == 99))
        self.assertTrue(torch.equal(perturb(ids, prefix, 99, 0, 1)[0, :3], ids[0, :3]))

    def test_loss_ignores_infinite_padding_and_has_correct_gradient(self):
        current = torch.tensor([[-1., -float('inf')], [-1., -float('inf')]], requires_grad=True)
        old = current.detach().clone()
        valid = torch.tensor([[True, False], [True, False]])
        loss, stats = clipped_loss(current, old, old, torch.tensor([1., -1.]), valid, .2, .04)
        loss.backward()
        self.assertLess(current.grad[0, 0], 0)
        self.assertGreater(current.grad[1, 0], 0)
        self.assertEqual(stats['kl'], 0)
        self.assertTrue(torch.isfinite(current.grad).all())

    def test_clipping_stops_positive_advantage_above_upper_bound(self):
        current = torch.tensor([[0.]], requires_grad=True)
        old = torch.tensor([[-1.]])
        loss, _ = clipped_loss(current, old, current.detach(), torch.tensor([1.]),
                               torch.tensor([[True]]), .2, 0)
        loss.backward()
        self.assertEqual(current.grad.item(), 0)

    def test_microbatch_weighting_preserves_group_loss_and_gradient(self):
        current = torch.tensor([[-1., -2.], [-3., -4.]], requires_grad=True)
        old = current.detach() - .1
        ref = current.detach() + .2
        valid = torch.tensor([[True, True], [True, False]])
        advantages = torch.tensor([.5, -.5])
        full, stats = clipped_loss(current, old, ref, advantages, valid, .2, .04)
        self.assertAlmostEqual(full.item(), stats['policy_loss'] + stats['kl_penalty'], places=6)
        gradient = torch.autograd.grad(full, current)[0]
        split = 0
        for i in range(2):
            loss, _ = clipped_loss(current[i:i+1], old[i:i+1], ref[i:i+1],
                                    advantages[i:i+1], valid[i:i+1], .2, .04)
            split = split + loss * valid[i:i+1].sum() / valid.sum()
        torch.testing.assert_close(split, full)
        torch.testing.assert_close(torch.autograd.grad(split, current)[0], gradient)

    def test_reference_restores_parameters_even_on_error(self):
        model = torch.nn.Module()
        model.lora_B = torch.nn.Parameter(torch.ones(2))
        with self.assertRaises(RuntimeError):
            with reference_adapter(model):
                self.assertTrue(torch.equal(model.lora_B, torch.zeros(2)))
                raise RuntimeError('test')
        self.assertTrue(torch.equal(model.lora_B, torch.ones(2)))


if __name__ == '__main__':
    unittest.main()
