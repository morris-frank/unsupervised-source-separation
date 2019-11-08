import numpy as np
import torch


def dilate(x: torch.Tensor, target_dilation: int) -> torch.Tensor:
    prev_dilation, c, l = x.size()  # n: prev dilation, c: num of channels, l: input length

    if prev_dilation == target_dilation:
        # Already have target size, nothing to dilate
        return x

    dilation_factor = target_dilation / prev_dilation

    new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
    if new_l != l:
        x = pad1d(x, new_l, dim=2)

    # reshape according to dilation
    x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    x = x.view(c, new_l, target_dilation)
    x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return x


def pad1d(x: torch.Tensor, new_l: int, dim: int) -> torch.Tensor:
    # TODO
    return x
