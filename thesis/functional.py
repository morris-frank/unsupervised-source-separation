from itertools import chain
from typing import Tuple

import torch
from torch import Tensor as T
from torch.nn import functional as F


def shift1d(tensor: T, shift: int) -> T:
    """
    Shifts a Tensor to the left or right and pads with zeros.

    Args:
        tensor: Input Tensor [N×C×L]
        shift: the shift, negative for left shift, positive for right

    Returns:
        the shifted tensor, same size
    """
    assert tensor.ndimension() == 3
    L = tensor.shape[2]
    pad = [-min(shift, 0), max(shift, 0)]
    out = F.pad(tensor, pad)
    out = out[:, :, pad[1] : pad[1] + L]
    return out.contiguous()


def discretize(tensor: T, μ: int = 101):
    assert μ & 1
    assert tensor.max() <= 1.0 and tensor.min() >= -1.0
    μ -= 1
    hμ = μ // 2
    out = torch.round(tensor * hμ) + hμ
    return out


def normalize(waveform: T):
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


def interleave(tensors: Tuple[T, ...], groups: int, dim: int = 1) -> T:
    cs = tensors[0].shape[dim] // groups
    splits = map(lambda x: torch.split(x, cs, dim=dim), tensors)
    return torch.cat(list(chain.from_iterable(zip(*splits))), dim=dim)


def chunk(tensor: T, chunks: int = 2, groups: int = 1, dim: int = 1) -> Tuple[T, ...]:
    cs = tensor.shape[dim] // (groups * chunks)  # Size of one chunk
    splits = tensor.split(cs, dim=dim)
    chunks = tuple(torch.cat(splits[i::chunks], dim=dim) for i in range(chunks))
    return chunks


def permute_L2C(x: T, factor: int = 2) -> T:
    N, C, L = x.shape
    squeezed = x.view(N, C, L // factor, factor).permute(0, 1, 3, 2)
    out = squeezed.contiguous().view(N, C * factor, L // factor)
    return out


def permute_C2L(x: T, factor: int = 2) -> T:
    N, C, L = x.shape
    out = x.view(N, C // factor, factor, L).permute(0, 1, 3, 2)
    out = out.contiguous().view(N, C // factor, L * factor)
    return out


def flip(tensor: T, chunks: int = 2, groups: int = 1, dim: int = 1) -> T:
    up, down = chunk(tensor, chunks=chunks, groups=groups, dim=dim)
    return interleave((down, up), groups=groups, dim=dim)
