"""MDLM diffu-GRPO proxy objective; not a diffusion trajectory likelihood."""
from contextlib import contextmanager

import torch


def completion_mask(tokens, prefix_lengths, eos_id, pad_id, mask_id):
    positions = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
    suffix = positions >= prefix_lengths[:, None]
    eos = suffix & tokens.eq(eos_id)
    first_eos = torch.where(eos, positions, tokens.shape[1]).amin(1)
    return (suffix & (positions <= first_eos[:, None])
            & tokens.ne(pad_id) & tokens.ne(mask_id))


def grouped_advantages(rewards, group_size):
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError('Rewards must contain complete groups of at least two.')
    groups = rewards.reshape(-1, group_size)
    return (groups - groups.mean(1, keepdim=True)).reshape(-1)


def perturb(tokens, prefix_lengths, mask_id, prompt_probability, seed):
    if not 0 <= prompt_probability <= 1:
        raise ValueError('prompt_probability must be in [0, 1].')
    positions = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
    prompt = positions < prefix_lengths[:, None]
    generator = torch.Generator(device=tokens.device).manual_seed(seed)
    hidden = ~prompt | (torch.rand(tokens.shape, device=tokens.device,
                                  generator=generator) < prompt_probability)
    return tokens.masked_fill(hidden, mask_id)


def partial_completion_mask(valid, probability, seed):
    """Fixed-size random subset per candidate; deterministic across PPO reuse."""
    if not 0 < probability <= 1:
        raise ValueError('Completion mask probability must be in (0, 1].')
    if probability == 1:
        return valid.clone()
    generator = torch.Generator(device=valid.device).manual_seed(seed)
    selected = torch.zeros_like(valid)
    for i in range(len(valid)):
        positions = valid[i].nonzero().flatten()
        if positions.numel():
            n = max(1, round(probability * positions.numel()))
            order = torch.randperm(len(positions), device=valid.device, generator=generator)
            selected[i, positions[order[:n]]] = True
    return selected


def proxy_logps(model, tokens, prefix_lengths, prompt_probability, seed,
                completion_probability=1.):
    xt = perturb(tokens, prefix_lengths, model.mask_index, prompt_probability, seed)
    if completion_probability < 1:
        valid = completion_mask(tokens, prefix_lengths, model.tokenizer.eos_token_id,
                                model.tokenizer.pad_token_id, model.mask_index)
        selected = partial_completion_mask(valid, completion_probability, seed)
        # Keep PAD/post-EOS tail hidden, reveal only valid unselected completion.
        xt = torch.where(valid & ~selected, tokens, xt)
    t = torch.ones(tokens.shape[0], 1, device=tokens.device)
    _, alpha = model.noise(t)
    sigma = model._sigma_from_alphat(alpha)
    logps = model.forward(xt=xt, sigma=sigma)
    return logps.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


@contextmanager
def reference_adapter(model):
    """Zero LoRA B temporarily; frozen imported EMA weights define reference."""
    parameters = [p for n, p in model.named_parameters() if n.endswith('lora_B')]
    if not parameters:
        raise ValueError('Reference adapter requires a LoRA model.')
    saved = [p.detach().clone() for p in parameters]
    try:
        with torch.no_grad():
            for p in parameters:
                p.zero_()
        yield
    finally:
        with torch.no_grad():
            for p, value in zip(parameters, saved):
                p.copy_(value)


def clipped_loss(current, old, reference, advantages, valid, epsilon, beta):
    # Select before arithmetic: ignored MASK/PAD logps may be -infinity.
    current, old, reference = current[valid], old[valid], reference[valid]
    if current.numel() == 0:
        raise ValueError('No valid completion tokens.')
    advantage = advantages[:, None].expand_as(valid)[valid]
    ratio = (current - old).exp()
    surrogate = torch.minimum(ratio * advantage,
                              ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)
    delta = reference - current
    kl = delta.exp() - delta - 1
    loss = (-surrogate + beta * kl).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite GRPO loss; update aborted.')
    return loss, {'policy_loss': -surrogate.detach().mean().item(),
                  'kl_penalty': beta * kl.detach().mean().item(),
                  'kl': kl.detach().mean().item(),
                  'clip_fraction': ((ratio - 1).abs() > epsilon).float().mean().item()}
