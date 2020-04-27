from itertools import chain
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


def normalize(waveform: torch.Tensor):
    """
    Removes any DC offset and scales to range [-1, 1]

    Args:
        waveform:

    Returns:
        scaled waveform
    """
    waveform = waveform - waveform.mean(dim=-1, keepdim=True)
    waveform = F.normalize(waveform, p=float("inf"), dim=-1)
    return waveform


def split(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return x[:, :, ::2].contiguous(), x[:, :, 1::2].contiguous()


def interleave(tensors: Tuple[torch.Tensor], groups: int, dim: int = 1):
    cs = tensors[0].shape[dim] // groups
    splits = map(lambda x: torch.split(x, cs, dim=dim), tensors)
    return torch.cat(list(chain.from_iterable(zip(*splits))), dim=dim)


def chunk_grouped(tensor: torch.Tensor, groups: int, dim: int = 1):
    #  Right now only outputs 2 chunks!
    cs = tensor.shape[dim] // (groups * 2)  # Size of one chunk
    chunks = tensor.split(cs, dim=dim)
    return torch.cat(chunks[::2], dim=dim), torch.cat(chunks[1::2], dim=dim)


def split_LtoC(x: torch.Tensor) -> torch.Tensor:
    N, C, L = x.shape
    squeezed = x.view(N, C, L // 2, 2).permute(0, 1, 3, 2)
    out = squeezed.contiguous().view(N, C * 2, L // 2)
    return out


def split_CtoL(x: torch.Tensor) -> torch.Tensor:
    N, C, L = x.shape
    out = x.view(N, C // 2, 2, L).permute(0, 1, 3, 2)
    out = out.contiguous().view(N, C // 2, L * 2)
    return out


def flip_channels(x: torch.Tensor) -> torch.Tensor:
    x_a, x_b = x.chunk(2, 1)
    return torch.cat([x_b, x_a], 1)
