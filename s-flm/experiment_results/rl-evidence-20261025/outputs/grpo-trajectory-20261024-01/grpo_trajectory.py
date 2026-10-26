"""Read-only statistics for replaying fixed absorbing-diffusion transitions."""
import torch


def transition_masks(before,after,prefix_length,mask_id):
    suffix=torch.arange(before.numel(),device=before.device)>=prefix_length
    if not torch.equal(before[:prefix_length],after[:prefix_length]):
        raise ValueError('Prefix changed.')
    if not torch.equal(before[before!=mask_id],after[before!=mask_id]):
        raise ValueError('An absorbing transition changed a revealed token.')
    masked=suffix & before.eq(mask_id)
    revealed=masked & after.ne(mask_id)
    remained=masked & after.eq(mask_id)
    return masked,revealed,remained


def kernel_log_probability(target_logps,revealed,remained,denoise_probability):
    if not 0 < denoise_probability <= 1:raise ValueError('Invalid denoise probability.')
    if denoise_probability==1 and remained.any():raise ValueError('Final transition retained MASK.')
    h=torch.tensor(denoise_probability,dtype=torch.float64,device=target_logps.device)
    value=target_logps[revealed].double().sum()+revealed.sum()*h.log()
    if remained.any():value=value+remained.sum()*torch.log1p(-h)
    return value.item()
