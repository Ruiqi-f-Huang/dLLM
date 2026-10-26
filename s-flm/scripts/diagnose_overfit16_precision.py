"""FP32 control for a tiny code-only diagnostic update; never saves weights."""
import gc
import inspect
import json
import sys
import textwrap
import types
from pathlib import Path

import hydra
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as entry
import algo
import dataloader
from scripts.diagnose_overfit16_gradients import tensors, forward, measure, norm, frozen_hash


@hydra.main(version_base=None, config_path='../configs', config_name='sft-overfit16')
def run(config):
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    p = Path('/root/autodl-tmp/s-flm/outputs/overfit16-20261023-01')
    rows = [json.loads(s) for s in Path(config.data.train_file).read_text().splitlines()]
    tokenizer = dataloader.get_tokenizer(config)
    data = tensors(rows, tokenizer, str(config.data.separator))
    result = {'precision': 'FP32, TF32 off; backbone autocast disabled on this instance only', 'models': {}}
    for step in [0, 3000]:
        if step == 0:
            model = entry._initialize_finetune(algo.MDLM, config, tokenizer)
        else:
            model = algo.MDLM(config, tokenizer=tokenizer)
            ckpt = torch.load(p / 'checkpoints/step-003000.ckpt', map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['state_dict'], strict=True)
            del ckpt
        model.ema = None
        model.to('cuda').eval()
        # Rebind a copy of the original forward, changing only autocast enablement.
        original = type(model.backbone).forward
        source = textwrap.dedent(inspect.getsource(original))
        assert source.count('enabled=x.is_cuda') == 1
        local = {}
        exec(source.replace('enabled=x.is_cuda', 'enabled=False'), original.__globals__, local)
        model.backbone.forward = types.MethodType(local['forward'], model.backbone)
        before = measure(model, data, rows)
        assert before == measure(model, data, rows)
        digest = frozen_hash(model)
        params = [v for v in model.parameters() if v.requires_grad]
        initial = [v.detach().clone() for v in params]
        optimizer = torch.optim.AdamW(params, lr=float(config.optim.lr),
                                     betas=(float(config.optim.beta1), float(config.optim.beta2)),
                                     eps=float(config.optim.eps), weight_decay=float(config.optim.weight_decay))
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(rows), 4):
            batch = tuple(x[start:start + 4] for x in data)
            ce, _, _, _, _, _ = forward(model, batch, 1., rows[start:start + 4])
            (ce[batch[3].to('cuda') == 2].sum() / int((data[3] == 2).sum())).backward()
        norm_before = float(torch.nn.utils.clip_grad_norm_(params, float(config.trainer.gradient_clip_val)))
        optimizer.step()
        changes = [v.detach() - old for v, old in zip(params, initial)]
        stats = {'before': before, 'after': measure(model, data, rows),
                 'grad_norm_pre_clip': norm_before, 'lr': float(config.optim.lr),
                 'update_l2': norm(torch.cat([v.flatten() for v in changes])),
                 'frozen_parameters_unchanged': digest == frozen_hash(model)}
        assert stats['frozen_parameters_unchanged']
        result['models'][str(step)] = stats
        print(json.dumps({'step': step, **stats}), flush=True)
        del model, optimizer, params, initial, changes
        gc.collect()
        torch.cuda.empty_cache()
    (p / 'gradient-diagnostic/fp32-control.json').write_text(json.dumps(result, indent=2))
    print('FP32_CONTROL_COMPLETE', flush=True)


if __name__ == '__main__':
    run()
