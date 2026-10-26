"""Small native LoRA implementation for the repository's DiT linear layers."""

import fnmatch
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


LORA_PARAMETER_SUFFIXES = ("lora_A", "lora_B")


class LoRALinear(nn.Linear):
  """A state-dict-compatible Linear layer with a low-rank residual branch."""

  def __init__(self, in_features, out_features, bias=True, *, rank, alpha,
               dropout=0.0, device=None, dtype=None):
    if rank <= 0:
      raise ValueError("LoRA rank must be positive")
    if alpha <= 0:
      raise ValueError("LoRA alpha must be positive")
    if not 0 <= dropout < 1:
      raise ValueError("LoRA dropout must be in [0, 1)")
    super().__init__(in_features, out_features, bias=bias,
                     device=device, dtype=dtype)
    self.rank = int(rank)
    self.alpha = float(alpha)
    self.scaling = self.alpha / self.rank
    self.lora_dropout = nn.Dropout(float(dropout))
    self.lora_A = nn.Parameter(torch.empty(
      self.rank, in_features, device=device, dtype=dtype))
    self.lora_B = nn.Parameter(torch.zeros(
      out_features, self.rank, device=device, dtype=dtype))
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

  @classmethod
  def from_linear(cls, linear, *, rank, alpha, dropout):
    result = cls(
      linear.in_features, linear.out_features,
      bias=linear.bias is not None, rank=rank, alpha=alpha,
      dropout=dropout, device=linear.weight.device,
      dtype=linear.weight.dtype)
    with torch.no_grad():
      result.weight.copy_(linear.weight)
      if linear.bias is not None:
        result.bias.copy_(linear.bias)
    return result

  def forward(self, inputs):
    base = F.linear(inputs, self.weight, self.bias)
    update = F.linear(F.linear(self.lora_dropout(inputs), self.lora_A),
                      self.lora_B)
    return base + update * self.scaling


def is_lora_parameter_name(name):
  return name.endswith(LORA_PARAMETER_SUFFIXES)


def _set_child(parent, name, child):
  if isinstance(parent, (nn.Sequential, nn.ModuleList)):
    parent[int(name)] = child
  else:
    setattr(parent, name, child)


def apply_lora(module, *, target_modules, rank, alpha, dropout):
  """Replace matching Linear children and freeze everything except adapters."""
  targets = tuple(str(name) for name in target_modules)
  if not targets:
    raise ValueError("LoRA target_modules must not be empty")
  replacements = []
  for name, child in list(module.named_modules()):
    if not isinstance(child, nn.Linear) or isinstance(child, LoRALinear):
      continue
    if not any(fnmatch.fnmatchcase(name, target)
               or name == target or name.endswith(f".{target}")
               for target in targets):
      continue
    parent_name, _, child_name = name.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    replacement = LoRALinear.from_linear(
      child, rank=rank, alpha=alpha, dropout=dropout)
    _set_child(parent, child_name, replacement)
    replacements.append(name)
  if not replacements:
    raise ValueError(f"No Linear modules matched LoRA targets: {targets}")

  for parameter in module.parameters():
    parameter.requires_grad = False
  for name, parameter in module.named_parameters():
    if is_lora_parameter_name(name):
      parameter.requires_grad = True
  return replacements


def parameter_counts(module):
  total = sum(parameter.numel() for parameter in module.parameters())
  trainable = sum(parameter.numel() for parameter in module.parameters()
                  if parameter.requires_grad)
  return {"total": total, "trainable": trainable}
