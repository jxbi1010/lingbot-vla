import torch
import torch.nn as nn


def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    """EMA: ``dest = rate * dest + (1 - rate) * src``. Accumulates in float32, writes back in ``dest`` dtype so mixed-precision / eval forward cannot break updates."""
    param_dict_src = dict(model_src.named_parameters())
    with torch.no_grad():
        for _p_name, p_dest in model_dest.named_parameters():
            p_src = param_dict_src[_p_name]
            assert p_src is not p_dest
            dest_dtype = p_dest.dtype
            updated = p_dest.data.float().mul_(rate).add_(p_src.data.float().mul(1 - rate))
            p_dest.data.copy_(updated.to(dtype=dest_dtype))