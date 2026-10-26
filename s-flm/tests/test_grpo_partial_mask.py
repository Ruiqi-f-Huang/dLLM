import unittest
import torch
from grpo_partial_mask import make_mask,noisy_input,question_scores,paired_summary


class PartialMaskTests(unittest.TestCase):
    def test_nested_exact_masks_and_global_rng_unchanged(self):
        eligible=torch.tensor([False]*3+[True]*20+[False]*4)
        state=torch.random.get_rng_state()
        a=make_mask(eligible,.25,'q',1,0);b=make_mask(eligible,.75,'q',1,0)
        self.assertEqual(int(a.sum()),5);self.assertEqual(int(b.sum()),15)
        self.assertFalse((a & ~b).any());self.assertFalse((b & ~eligible).any())
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(make_mask(eligible,1,'q',1,0),eligible))

    def test_prefix_and_tail_context_policy(self):
        tokens=torch.arange(10)
        eligible=torch.tensor([False,False,True,True,True,True,False,False,False,False])
        masked=make_mask(eligible,.5,'q',0,1)
        noisy=noisy_input(tokens,2,eligible,masked,99)
        self.assertTrue(torch.equal(noisy[:2],tokens[:2]))
        self.assertTrue(torch.equal(noisy[eligible & ~masked],tokens[eligible & ~masked]))
        self.assertTrue((noisy[masked]==99).all());self.assertTrue((noisy[6:]==99).all())
        full=noisy_input(tokens,2,eligible,eligible,99)
        self.assertTrue((full[2:]==99).all())

    def test_replicates_are_averaged_before_question_comparison(self):
        rows=[]
        for model,shift in [('base',0),('rl',.25)]:
            for sid in ['a','b']:
                for rep,value in enumerate([-2.,-1.]):
                    rows.append(dict(model=model,rate=.5,reward=1,region='code_eos',sample_id=sid,
                        tokens=3,mean_logp=value+shift,top1=.5,replicate=rep))
        q=question_scores(rows)
        result=paired_summary([x for x in q if x['model']=='base'],[x for x in q if x['model']=='rl'])
        self.assertEqual(result['questions'],2)
        self.assertEqual(result['mean_logp_delta'],.25)
        self.assertEqual(result['ce_delta'],-.25)
        self.assertEqual(result['mean_logp_ci95'],[.25,.25])
        self.assertEqual(q[0]['mask_replicates'],2)

    def test_zero_region_is_not_scored_as_zero_ce(self):
        self.assertEqual(question_scores([dict(tokens=0)]),[])
        self.assertEqual(paired_summary([],[]),dict(questions=0))
        with self.assertRaises(ValueError):
            paired_summary([dict(sample_id='a')],[])


if __name__=='__main__':unittest.main()
