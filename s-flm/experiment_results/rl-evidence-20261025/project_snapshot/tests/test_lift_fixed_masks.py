import unittest

import torch

from lift_masking import code_region
from scripts.diagnose_lift_fixed_masks import critical_regions, conditions, fixed_uniforms


class LiftFixedMaskTests(unittest.TestCase):
    def test_local_variables_exclude_builtins_and_docstring_contents(self):
        code = 'def f():\n    """小明 123 + x"""\n    x = max(2, 3)\n    return x / 4\n'
        offsets = [(i, i + 1) for i in range(len(code))]
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {'input_ids':list(range(len(text))), 'offset_mapping':offsets}
        _, pure, ok = code_region(code, Tokenizer())
        self.assertTrue(ok)
        groups = critical_regions(code, offsets, pure)
        self.assertEqual(sum(groups['variable']), 2)
        self.assertEqual(sum(groups['number']), 3)
        self.assertEqual(sum(groups['operator']), 1)
        self.assertFalse(any(groups['variable'][code.index('max'):code.index('max')+3]))
        self.assertFalse(any(groups['critical'][code.index('"""'):code.rindex('"""')+3]))
        self.assertEqual(sum(groups['critical']), 6)

    def test_masks_are_reproducible_nested_and_prompt_stays_visible(self):
        length = 512
        torch.testing.assert_close(fixed_uniforms('one',17,length),fixed_uniforms('one',17,length))
        self.assertFalse(torch.equal(fixed_uniforms('one',17,length),fixed_uniforms('one',29,length)))
        suffix = (torch.arange(length) >= 23)[None,:]
        data = {'ids':torch.zeros(1,length,dtype=torch.long),'suffix':suffix,'sample_ids':['one']}
        masks = list(conditions(data))
        self.assertEqual(len(masks),9)
        previous = torch.zeros_like(suffix)
        for rate, seed, mask in masks:
            self.assertFalse(mask[:,:23].any())
            if seed == 17:
                self.assertFalse((previous & ~mask).any())
                previous = mask
            if rate == 1:
                torch.testing.assert_close(mask,suffix)


if __name__ == '__main__':
    unittest.main()
