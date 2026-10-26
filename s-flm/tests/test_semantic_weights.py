import unittest
import torch
from semantic_weights import critical_spans, weighted_mean


class SemanticWeightTests(unittest.TestCase):
    def test_docstring_unicode_and_computation(self):
        code = 'def solution():\n    """题目 has 99 and x + y"""\n    x = 3\n    y = x * 2\n    return y\n'
        spans = critical_spans(code)
        marked = {code[a:b] for a, b in spans}
        self.assertTrue({'x', 'y', '3', '2', '*'} <= marked)
        self.assertNotIn('99', marked)
        self.assertNotIn('+', marked)
        self.assertTrue(all(a >= code.index('    x =') for a, b in spans))

    def test_return_builtin_does_not_mark_type_annotation(self):
        code = 'def solution() -> int:\n    x = 3\n    return int(x)'
        spans = critical_spans(code)
        self.assertTrue(any(code[a:b] == 'int' for a, b in spans))
        self.assertTrue(all(a > code.index('\n') for a, b in spans))

    def test_normalization_and_visible_tokens(self):
        loss = torch.tensor([[0., 2., 6.]], requires_grad=True)
        valid = torch.tensor([[0, 1, 1]])
        self.assertEqual(weighted_mean(loss, valid, torch.ones_like(loss)).item(), 4.)
        self.assertEqual(weighted_mean(loss, valid, torch.tensor([[9., 1., 2.]])).item(), torch.tensor(14 / 3).item())
        weighted_mean(loss, valid, torch.ones_like(loss)).backward()
        self.assertEqual(loss.grad.tolist(), [[0., .5, .5]])

    def test_training_only_weighting_preserves_unweighted_nll_and_validation(self):
        from types import SimpleNamespace
        from trainer_base import TrainerBase
        model = SimpleNamespace(config={'noise': None}, log=lambda *a, **kw: None,
            _process_model_input=lambda x, mask: (x, None, mask),
            nll=lambda *a, **kw: (torch.tensor([[99., 2., 6.]]), None))
        x = torch.zeros(1, 3, dtype=torch.long)
        mask = torch.tensor([[0, 1, 1]])
        weights = torch.tensor([[1., 1., 2.]])
        base = TrainerBase._loss(model, x, mask, train_mode=True)
        uniform = TrainerBase._loss(model, x, mask, train_mode=True,
                                   loss_weights=torch.ones_like(weights))
        weighted = TrainerBase._loss(model, x, mask, train_mode=True, loss_weights=weights)
        validation = TrainerBase._loss(model, x, mask, loss_weights=weights)
        torch.testing.assert_close(base.loss, uniform.loss, rtol=0, atol=0)
        torch.testing.assert_close(weighted.nlls, base.nlls, rtol=0, atol=0)
        torch.testing.assert_close(weighted.num_tokens, base.num_tokens, rtol=0, atol=0)
        torch.testing.assert_close(validation.loss, base.loss, rtol=0, atol=0)
        self.assertAlmostEqual(weighted.loss.item(), 14 / 3, places=5)
