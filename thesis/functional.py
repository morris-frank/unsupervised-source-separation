import random
from math import log
from typing import Tuple

import torch
from torch.nn import functional as F


def dilate(x: torch.Tensor, new: int, old: int = 1) -> torch.Tensor:
    """
    Will dilate the input tensor of shape [N, C, L]

    Args:
        x: input tensor
        new: new amount of dilation to get
        old: current amount if dilation of tensor

    Returns:
        dilated tensor
    """
    n, c, w = x.shape  # N == Batch size × old
    dilation = new / old
    if dilation == 1:
        return x
    w, n = int(w / dilation), int(n * dilation)
    x = x.permute(1, 2, 0)
    x = torch.reshape(x, [c, w, n])
    x = x.permute(2, 0, 1)
    return x.contiguous()


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
    assert x.max() <= 1. and x.min() >= -1.
    μ -= 1
    hμ = μ // 2
    out = torch.round(x * hμ) + hμ
    return out


def encode_μ_law(x: torch.Tensor, μ: int = 255) -> torch.Tensor:
    """
    Encodes the input tensor element-wise with μ-law encoding

    Args:
        x: tensor
        μ: the size of the encoding (number of possible classes)

    Returns:
        the encoded tensor
    """
    assert μ & 1
    assert x.max() <= 1.0 and x.min() >= -1.0
    μ -= 1
    hμ = μ // 2
    out = torch.sign(x) * torch.log(1 + μ * torch.abs(x)) / log(μ)
    out = torch.round(out * hμ) + hμ
    return out


def decode_μ_law(x: torch.Tensor, μ: int = 255) -> torch.Tensor:
    """
    Applies the element-wise inverse μ-law encoding to the tensor.

    Args:
        x: input tensor
        μ: size of the encoding (number of possible classes)

    Returns:
        the decoded tensor
    """
    assert μ & 1
    μ = μ - 1
    hμ = μ // 2
    out = (x.type(torch.float32) - hμ) / hμ
    out = torch.sign(out) / μ * (torch.pow(μ, torch.abs(out)) - 1)
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


def orthonormal(*dim: int) -> torch.Tensor:
    """
    Creates an orthonormal Tensor with the given dimensions.

    Args:
        *dim: The dimensions.

    Returns:
        the Tensor (new)
    """
    # this guarantees |det(weights)| == 1 but not the sign
    data = torch.qr(torch.empty(*dim).normal_())[0]

    if torch.det(data) < 0:
        data[:, 0] = -data[:, 0]
    return data


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
