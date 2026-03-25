import torch
import torch.nn as nn


def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    """EMA: ``dest = rate * dest + (1 - rate) * src``.

    Uses in-place ``mul_`` / ``add_(..., alpha=)`` on ``param.data`` so FSDP2 ``DTensor``
    parameters stay in the distributed tensor path. Do not call ``.float()`` here: that
    produces plain tensors and breaks ``aten.add_.Tensor`` with mixed Tensor/DTensor.
    """
    param_dict_src = dict(model_src.named_parameters())
    with torch.no_grad():
        for _p_name, p_dest in model_dest.named_parameters():
            p_src = param_dict_src[_p_name]
            assert p_src is not p_dest
            d = p_dest.data
            s = p_src.data
            d.mul_(rate)
            d.add_(s, alpha=(1.0 - rate))
