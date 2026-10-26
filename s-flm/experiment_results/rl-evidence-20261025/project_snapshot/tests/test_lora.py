"""Native LoRA injection and compatibility checks."""

import unittest

import torch
import torch.nn as nn

from models.lora import LoRALinear, apply_lora, parameter_counts


class TinyBlock(nn.Module):
  def __init__(self):
    super().__init__()
    self.attn_qkv = nn.Linear(4, 12, bias=False)
    self.attn_out = nn.Linear(4, 4, bias=False)
    self.mlp = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))
    self.untouched = nn.Linear(4, 4)

  def forward(self, inputs):
    return self.attn_out(inputs) + self.mlp(inputs) + self.untouched(inputs)


class LoRATests(unittest.TestCase):
  def test_zero_initialized_adapter_preserves_output_and_freezes_base(self):
    torch.manual_seed(1)
    model = TinyBlock()
    inputs = torch.randn(2, 4)
    expected = model(inputs).detach()
    names = apply_lora(
      model, target_modules=['attn_qkv', 'attn_out', 'mlp.0', 'mlp.2'],
      rank=2, alpha=4, dropout=0)
    self.assertEqual(names, ['attn_qkv', 'attn_out', 'mlp.0', 'mlp.2'])
    torch.testing.assert_close(model(inputs), expected)
    self.assertTrue(all(
      parameter.requires_grad == name.endswith(('lora_A', 'lora_B'))
      for name, parameter in model.named_parameters()))
    self.assertEqual(parameter_counts(model)['trainable'], 96)

  def test_optimizer_changes_adapter_but_not_base(self):
    torch.manual_seed(2)
    model = TinyBlock()
    apply_lora(model, target_modules=['attn_out'], rank=2, alpha=4, dropout=0)
    base_before = model.attn_out.weight.detach().clone()
    adapter_before = model.attn_out.lora_B.detach().clone()
    optimizer = torch.optim.SGD(
      [parameter for parameter in model.parameters() if parameter.requires_grad],
      lr=.1)
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    torch.testing.assert_close(model.attn_out.weight, base_before)
    self.assertFalse(torch.equal(model.attn_out.lora_B, adapter_before))

  def test_base_state_dict_load_has_only_adapter_keys_missing(self):
    base = TinyBlock()
    state = base.state_dict()
    apply_lora(base, target_modules=['attn_out'], rank=2, alpha=4, dropout=0)
    incompatible = base.load_state_dict(state, strict=False)
    self.assertEqual(set(incompatible.missing_keys),
                     {'attn_out.lora_A', 'attn_out.lora_B'})
    self.assertEqual(incompatible.unexpected_keys, [])
    self.assertIsInstance(base.attn_out, LoRALinear)


if __name__ == '__main__':
  unittest.main()
