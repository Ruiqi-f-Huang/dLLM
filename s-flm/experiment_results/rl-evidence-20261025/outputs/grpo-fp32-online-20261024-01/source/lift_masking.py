"""LIFT mask selection. No changes to the MDLM loss or sampler.

LIFT: expanded candidate mask, gold-probability ranking, time argmax,
and restoring rejected positions. Adaptation: per-example q=1-alpha,
code/EOS eligibility, zero-k support, and isolated reproducible RNG.
"""
import ast

import torch


REGION_VERSION = 1


def code_region(code, tokenizer):
    """Exclude all Python docstrings, including BPEs crossing their boundary.

    Invalid Python or unreliable offsets return an all-false region. The
    caller must then use ordinary random masking for the entire example.
    AST columns are UTF-8 bytes; tokenizer offsets are character offsets.
    """
    try:
        encoded = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True)
    except (NotImplementedError, TypeError):
        encoded = tokenizer(code, add_special_tokens=False)
        return encoded['input_ids'], [False] * len(encoded['input_ids']), False
    ids, offsets = encoded['input_ids'], encoded.get('offset_mapping', [])
    try:
        tree = ast.parse(code)
        lines = code.splitlines(keepends=True)
        starts = [0]
        for line in lines:
            starts.append(starts[-1] + len(line))

        def position(line, column):
            return starts[line - 1] + len(
                lines[line - 1].encode('utf-8')[:column].decode('utf-8'))

        spans = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
                first = node.body[0]
                if (isinstance(first, ast.Expr)
                        and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    spans.append((position(first.lineno, first.col_offset),
                                  position(first.end_lineno, first.end_col_offset)))
        if (len(offsets) != len(ids)
                or any(not 0 <= a < b <= len(code) for a, b in offsets)):
            raise ValueError('Unreliable tokenizer offsets')
        region = [not any(a < end and b > start for start, end in spans)
                  for a, b in offsets]
        return ids, region, True
    except (SyntaxError, ValueError, UnicodeError, IndexError):
        return ids, [False] * len(ids), False


def prepare_masks(q, eligible, noise_mask, generator, draws=None):
    """Candidate and ordinary masks share draws outside the selected region."""
    if q.ndim != 2 or q.shape[1] != 1:
        raise ValueError('q must have shape [batch, 1]')
    if not torch.isfinite(q).all() or not ((q >= 0) & (q <= 1)).all():
        raise ValueError('Invalid mask probability')
    eligible = eligible.bool() & noise_mask.bool()
    upper = 1 - q
    lower = torch.where(upper <= .1, 0., .1)
    rho = lower + (upper - lower) * torch.rand(
        q.shape, device=q.device, generator=generator)
    if draws is None:
        draws = torch.rand(eligible.shape, device=q.device, generator=generator)
    ordinary = (draws < q) & noise_mask.bool()
    candidate = (draws < (q + rho).clamp(max=1)) & eligible
    outside = ordinary & ~eligible
    k = (candidate.sum(-1) * (q / (q + rho).clamp_min(1e-8))[:, 0]).round().long()
    k = torch.minimum(k.clamp_min(0), candidate.sum(-1))
    # Generate random ranking in both B and C so RNG consumption is identical.
    random_scores = torch.rand(eligible.shape, device=q.device, generator=generator)
    return candidate, outside, k, random_scores, rho


def select_mask(candidate, k, gold_scores, random_scores, q, policy):
    """Return selected positions and modes: random=0, easy=1, hard=2."""
    if policy not in ('random_restore', 'lift3'):
        raise ValueError(f'Unknown LIFT policy: {policy}')
    if policy == 'lift3':
        weights = torch.cat([3 * q * (1 - q), q, 1 - q], dim=1)
        modes = weights.argmax(dim=1)
    else:
        modes = torch.zeros(candidate.shape[0], device=q.device, dtype=torch.long)
    ranking = torch.where(modes[:, None] == 1, gold_scores,
                          torch.where(modes[:, None] == 2, -gold_scores, random_scores))
    ranking = ranking.masked_fill(~candidate, -torch.inf)
    # Stable ties make the selected set reproducible.
    order = ranking.argsort(dim=-1, descending=True, stable=True)
    take = torch.arange(candidate.shape[1], device=q.device)[None, :] < k[:, None]
    selected = torch.zeros_like(candidate).scatter(1, order, take)
    return selected & candidate, modes


def mdlm_lift_nll(self, x0, context, accumulation_step, noise_valid, region):
    """Code-region LIFT adaptation; evaluation always uses ordinary nll()."""

    t = self._sample_t(x0.shape[0], accumulation_step)
    dalpha, alpha = self.noise(t)
    alpha, dalpha = alpha[:, None], dalpha[:, None]
    q = 1 - alpha
    sigma = self._sigma_from_alphat(alpha)
    # Stateless per microbatch RNG: reproducible at checkpoint boundaries;
    # candidate selection cannot perturb time sampling or training dropout.
    generator = torch.Generator(device=x0.device)
    seed = (int(self.config.seed) + 1000003 * int(self.global_step)
                    + 1009 * int(accumulation_step or 0) + 9176 * int(self.global_rank))
    generator.manual_seed(seed % (2**63 - 1))
    eligible = region.bool() & noise_valid.bool()
    # Match the global RNG draw made by ordinary MDLM q_xt(). Time samples
    # and training dropout then stay aligned with A as well as between B/C.
    draws = torch.rand(*x0.shape, device=x0.device)
    candidate, outside, k, random_scores, rho = prepare_masks(
        q, eligible, noise_valid, generator, draws=draws)
    proposal = x0.masked_fill(candidate | outside, self.mask_index)
    devices = [x0.device.index] if x0.is_cuda else []
    # Both B/C execute scoring, but only C uses its probabilities. Fork RNG
    # restores dropout state before the trainable forward, in both groups.
    # Lightning wraps the whole step in AMP. Caching a no-grad cast of a
    # trainable weight here can otherwise detach the second forward too.
    # Preserve the surrounding precision, but never populate its cast cache.
    with (torch.random.fork_rng(devices=devices), torch.no_grad(),
          torch.autocast(x0.device.type,
                         enabled=torch.is_autocast_enabled(x0.device.type),
                         dtype=torch.get_autocast_dtype(x0.device.type),
                         cache_enabled=False)):
        scored = self.forward(x0=x0, xt=proposal, sigma=sigma, context=context)
        gold_scores = scored.gather(-1, x0[..., None]).squeeze(-1)
        del scored
    selected, modes = select_mask(candidate, k, gold_scores, random_scores,
                                  q, self.config.lift.policy)
    xt = x0.masked_fill(selected | outside, self.mask_index)
    log_probs = self.forward(x0=x0, xt=xt, sigma=sigma, context=context)
    loss = self.nll_per_token(log_probs, xt, x0, alpha, dalpha,
                              context=context, train_mode=True)
    with torch.no_grad():
        stats = {
            'rho': rho.mean(),
            'eligible_tokens': eligible.sum().float(),
            'candidate_tokens': candidate.sum().float(),
            'selected_tokens': selected.sum().float(),
            'actual_code_mask_rate': selected.sum() / eligible.sum().clamp_min(1),
            'target_code_mask_rate': ((q * eligible.sum(-1, keepdim=True)).sum()
                                     / eligible.sum().clamp_min(1)),
            'candidate_gold_logp': (gold_scores.masked_fill(~candidate, 0).sum()
                                   / candidate.sum().clamp_min(1)),
            'selected_gold_logp': (gold_scores.masked_fill(~selected, 0).sum()
                                  / selected.sum().clamp_min(1)),
            'region_fallback_fraction': (~eligible.any(-1)).float().mean(),
        }
        for idx, name in enumerate(('random', 'easy', 'hard')):
            stats[f'fraction_{name}'] = (modes == idx).float().mean()
        for name, value in stats.items():
            self.log(f'train/lift_{name}', value, on_step=True, on_epoch=False,
                     sync_dist=True)
    return loss, t

