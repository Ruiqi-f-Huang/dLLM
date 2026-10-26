"""Regression: three per-program timeouts must not exhaust the worker CPU budget."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_mdlm_grpo import execution_reward

@unittest.skipUnless(sys.platform=='linux','Production resource limits are tested on Linux')
class RewardTimeoutTests(unittest.TestCase):
    def test_nonterminating_program_is_model_failure_not_infrastructure_failure(self):
        result=execution_reward('def simple_math_problem():\n    while True:\n        pass', '#### 6',2.,details=True)
        self.assertFalse(result['infrastructure_failure'],result)
        self.assertEqual(result['reward'],0.)
        self.assertTrue(result['raw']['parses'])
        self.assertFalse(result['raw']['executes'])

if __name__=='__main__':unittest.main()
