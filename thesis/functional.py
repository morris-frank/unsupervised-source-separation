import random
from typing import Tuple

import torch
from torch.nn import functional as F


def shift1d(x: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Shifts a Tensor to the left or right and pads with zeros.

    Args:
        x: Input Tensor [N×C×L]
        shift: the shift, negative for left shift, positive for right

    Returns:
        the shifted tensor, same size
    """
    assert x.ndimension() == 3
    length = x.shape[2]
    pad = [-min(shift, 0), max(shift, 0)]
    y = F.pad(x, pad)
    y = y[:, :, pad[1] : pad[1] + length]
    return y.contiguous()


def discretize(x: torch.Tensor, μ: int = 101):
    assert μ & 1
    assert x.max() <= 1.0 and x.min() >= -1.0
    μ -= 1
    hμ = μ // 2
    out = torch.round(x * hμ) + hμ
    return out


def destroy_along_channels(x: torch.Tensor, amount: float) -> torch.Tensor:
    """
    Destroys a random subsample of channels in a Tensor ([N,C,L]).
    Destroy == set to 0.

    Args:
        x: Input tensor
        amount: percentage amount to destroy

    Returns:
        Destroyed tensor
    """
    if amount == 0:
        return x
    length = x.shape[1]
    for i in random.sample(range(length), int(amount * length)):
        if x.ndim == 3:
            x[:, i, :] = 0.0
        else:
            x[:, i] = 0.0
    return x


def split(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return x[:, :, ::2].contiguous(), x[:, :, 1::2].contiguous()


def interleave(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    assert left.shape == right.shape
    bs, c, l = left.shape
    return torch.stack((left, right), dim=3).view(bs, c, 2 * l)


def split_LtoC(x: torch.Tensor) -> torch.Tensor:
    N, C, L = x.shape
    squeezed = x.view(N, C, L // 2, 2).permute(0, 1, 3, 2)
    out = squeezed.contiguous().view(N, C * 2, L // 2)
    return out


def flip_channels(x: torch.Tensor) -> torch.Tensor:
    x_a, x_b = x.chunk(2, 1)
    return torch.cat([x_b, x_a], 1)
