"""Pure protocol helpers for the fixed-question GRPO diagnostic."""
import hashlib
import math
import random


def stable_seed(namespace, *parts):
    text = '|'.join(map(str, (namespace,) + parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'big') % (2**31)


def select_mixed_questions(rows, audit, count=32, seed=42):
    by_id = {r['sample_id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate training sample IDs.')
    rewards = {}
    seen = set()
    for record in audit:
        sid = record['sample_id']
        if sid not in by_id:
            raise ValueError(f'Audit contains a non-training question: {sid}')
        key = (sid, record['candidate'])
        if key in seen or record['reward'] not in (0, 1):
            raise ValueError('Duplicate audit candidates or non-binary rewards.')
        seen.add(key)
        rewards.setdefault(sid, set()).add(record['reward'])
    eligible = [sid for sid, values in rewards.items() if values == {0, 1}]
    eligible.sort(key=lambda sid: (stable_seed('selection', seed, sid), sid))
    if len(eligible) < count or count < 1:
        raise ValueError(f'Need {count} mixed-reward questions; found {len(eligible)}.')
    return [by_id[sid] for sid in eligible[:count]]


def balanced_indices(size, groups, seed):
    if size < 1 or groups < 1:
        raise ValueError('Positive schedule size required.')
    rng = random.Random(seed)
    result = []
    while len(result) < groups:
        cycle = list(range(size))
        rng.shuffle(cycle)
        result.extend(cycle[:groups-len(result)])
    return result


def summarize_samples(records, repeats):
    """Question-macro success probability and empirical at-least-one coverage."""
    grouped = {}
    for r in records:
        if r['reward'] not in (0, 1):
            raise ValueError('Expected binary execution rewards.')
        per_question = grouped.setdefault(r['sample_id'], {})
        if r['repeat'] in per_question:
            raise ValueError('Duplicate repeat index.')
        per_question[r['repeat']] = r
    if not grouped or any(set(v) != set(range(repeats)) for v in grouped.values()):
        raise ValueError('Each question needs the exact requested repeats.')
    per_question = []
    for sid, values in grouped.items():
        hits = sum(r['reward'] for r in values.values())
        per_question.append(dict(sample_id=sid, correct=hits, draws=repeats,
                                 success_probability=hits/repeats))
    mean = sum(q['success_probability'] for q in per_question)/len(per_question)
    return dict(questions=len(grouped), draws=len(records), repeats=repeats,
                success_at_1=mean, mean_reward=mean,
                empirical_pass_at_k=sum(q['correct'] > 0 for q in per_question)/len(grouped),
                per_question=per_question)


def paired_bootstrap(base, current, seed=31415, draws=2000):
    """Resample questions, preserving all repeated samples within each question."""
    a = {r['sample_id']: r['success_probability'] for r in base['per_question']}
    b = {r['sample_id']: r['success_probability'] for r in current['per_question']}
    if a.keys() != b.keys() or not a:
        raise ValueError('Unmatched question sets.')
    deltas = [b[k]-a[k] for k in sorted(a)]
    rng = random.Random(seed)
    values = sorted(sum(rng.choices(deltas, k=len(deltas)))/len(deltas) for _ in range(draws))
    return dict(mean_delta=sum(deltas)/len(deltas),
                question_bootstrap_ci95=[values[math.floor(.025*(draws-1))],
                                         values[math.ceil(.975*(draws-1))]],
                improved_questions=sum(d > 0 for d in deltas),
                worsened_questions=sum(d < 0 for d in deltas))
