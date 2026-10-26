import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_final_gsm8k_test import read_existing
from grpo_overfit import stable_seed

class ResumeValidationTests(unittest.TestCase):
    def test_accepts_partial_batch_but_rejects_mixed_protocol_or_bad_reward(self):
        rows=[dict(sample_id=str(i),question='q'+str(i),answer='a') for i in range(10)]
        records=[dict(**r,seed=2,batch_seed=stable_seed('final-gsm8k-test-v1',2,(i//8)*8),reward=0,execution={'infrastructure_failure':False}) for i,r in enumerate(rows)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'samples.jsonl'
            def write(rs):path.write_text(''.join(json.dumps(r)+'\n' for r in rs))
            write(records[:9]);self.assertEqual(len(read_existing(path,rows,2)),9)
            for key,value in [('seed',1),('batch_seed',123),('sample_id','wrong'),('question','different'),('reward',.5),('execution',{'infrastructure_failure':True})]:
                bad=[dict(r) for r in records[:9]];bad[-1][key]=value;write(bad)
                with self.subTest(key=key),self.assertRaises(ValueError):read_existing(path,rows,2)
            write(records+[records[-1]])
            with self.assertRaises(ValueError):read_existing(path,rows,2)

if __name__=='__main__':unittest.main()
