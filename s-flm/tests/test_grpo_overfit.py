import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from collections import Counter

from grpo_overfit import (balanced_indices, paired_bootstrap, select_mixed_questions,
                         stable_seed, summarize_samples)


class FixedQuestionTests(unittest.TestCase):
    def test_selection_is_order_independent_and_only_mixed(self):
        rows = [dict(sample_id=str(i)) for i in range(5)]
        audit = [dict(sample_id=str(i), candidate=j, reward=j if i < 4 else 1)
                 for i in range(5) for j in range(2)]
        a = select_mixed_questions(rows, audit, 3)
        b = select_mixed_questions(rows[::-1], audit[::-1], 3)
        self.assertEqual(a, b)
        self.assertNotIn('4', [r['sample_id'] for r in a])
        with self.assertRaises(ValueError):
            select_mixed_questions(rows, audit, 5)
        with self.assertRaises(ValueError):
            select_mixed_questions(rows, audit + [audit[0]], 3)

    def test_cycle_schedule_is_balanced_and_deterministic(self):
        schedule = balanced_indices(32, 250, 43)
        self.assertEqual(schedule, balanced_indices(32, 250, 43))
        self.assertEqual(len(schedule), 250)
        self.assertEqual(set(Counter(schedule).values()), {7, 8})
        self.assertEqual(set(schedule[:32]), set(range(32)))
        self.assertNotEqual(schedule[:32], schedule[32:64])

    def test_rng_namespaces_do_not_reuse_holdout_evaluation_seeds(self):
        self.assertNotEqual(stable_seed('holdout-proxy-pairs', 1, 'q', .3, 0),
                            stable_seed('fresh-evaluation', 1, 'q', .3, 0))
        self.assertEqual(stable_seed('fresh-evaluation', 1, 'q', .3, 0),
                         stable_seed('fresh-evaluation', 1, 'q', .3, 0))

    def test_success_probability_is_not_coverage(self):
        records = [dict(sample_id='a', repeat=i, reward=int(i==0)) for i in range(16)]
        records += [dict(sample_id='b', repeat=i, reward=0) for i in range(16)]
        s = summarize_samples(records, 16)
        self.assertEqual(s['success_at_1'], 1/32)
        self.assertEqual(s['mean_reward'], s['success_at_1'])
        self.assertEqual(s['empirical_pass_at_k'], .5)
        with self.assertRaises(ValueError):
            summarize_samples(records[:-1], 16)
        with self.assertRaises(ValueError):
            summarize_samples(records + [records[0]], 16)

    def test_prepare_cli_freezes_32_questions_and_provenance(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            train = [dict(sample_id=str(i), question=f'question {i}', answer='1') for i in range(33)]
            audit = [dict(sample_id=str(i), candidate=j, reward=j % 2)
                     for i in range(33) for j in range(8)]
            for name, rows in [('train', train), ('validation', [dict(sample_id='v', question='validation')]),
                               ('rollouts', audit)]:
                (path / f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (path / 'run.json').write_text(json.dumps(dict(audit_only=True, group_size=8, temperature=.3)))
            command = [sys.executable, str(root / 'scripts/grpo_fixed32.py'), 'prepare',
                       '--train', str(path / 'train.jsonl'), '--validation', str(path / 'validation.jsonl'),
                       '--audit', str(path / 'rollouts.jsonl'), '--output', str(path / 'fixed')]
            subprocess.run(command, check=True, capture_output=True)
            selected = [json.loads(x) for x in (path / 'fixed/questions.jsonl').read_text().splitlines()]
            metadata = json.loads((path / 'fixed/selection.json').read_text())
            self.assertEqual(len(selected), 32)
            self.assertEqual(metadata['ids'], [r['sample_id'] for r in selected])
            self.assertEqual(len(metadata['source_sha256']), 3)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_bootstrap_pairs_questions_not_individual_draws(self):
        a = dict(per_question=[dict(sample_id='a', success_probability=.25),
                               dict(sample_id='b', success_probability=.5)])
        b = dict(per_question=[dict(sample_id='b', success_probability=.75),
                               dict(sample_id='a', success_probability=.5)])
        result = paired_bootstrap(a, b)
        self.assertEqual(result['mean_delta'], .25)
        self.assertEqual(result['question_bootstrap_ci95'], [.25, .25])
        self.assertEqual(result['improved_questions'], 2)


if __name__ == '__main__':
    unittest.main()
