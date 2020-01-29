import math
import random

import torch
from torch.nn import functional as F


def dilate(x: torch.Tensor, new: int, old: int = 1) -> torch.Tensor:
    """
    Will dilate the input tensor of shape [N, C, L]

    Args:
        x: input tensor
        new: new amount of dilation to get
        old: current amount if dilation of x

    Returns:
        dilated x
    """
    [n, c, l] = x.shape  # N == Batch size × old
    dilation = new / old
    if dilation == 1:
        return x
    l, n = int(l / dilation), int(n * dilation)
    x = x.permute(1, 2, 0)
    x = torch.reshape(x, [c, l, n])
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
    y = y[:, :, pad[1]:pad[1] + length]
    return y.contiguous()


def encode_μ_law(x: torch.Tensor, μ: int = 255) -> torch.Tensor:
    """
    Encodes the input tensor element-wise with μ-law encoding

    Args:
        x: tensor
        μ: the size of the encoding (number of possible classes)

    Returns:
        the encoded tensor
    """
    assert x.max() <= 1. and x.min() >= -1.
    out = torch.sign(x) * torch.log(1 + μ * torch.abs(x)) / math.log(1 + μ)
    out = torch.floor(out * math.ceil(μ / 2))
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
    x = x.type(torch.float32)
    # out = (x + 0.5) * 2. / (μ + 1)
    out = x / math.ceil(μ / 2)
    out = torch.sign(out) / μ * (torch.pow(1 + μ, torch.abs(out)) - 1)
    return out


def destroy_along_channels(x: torch.Tensor, amount: float) -> torch.Tensor:
    """
    Destroys a random subsample of channels in a Tensor ([N,C,L]).
    Destroy == set to 0.

    Args:
        x: Input tensor
        amount: percentage amount to destroy

    Returns:
        Destroyed x
    """
    if amount == 0:
        return x
    length = x.shape[1]
    for i in random.sample(range(length), int(amount * length)):
        if x.ndim == 3:
            x[:, i, :] = 0.
        else:
            x[:, i] = 0.
    return x


def multi_argmax(x: torch.Tensor, n: int, μ: int = 101):
    """
    Takes argmax from SoftMax-output over the concatenated channels.

    Args:
        x: Output of network pred
        n: Number of sources
        μ: Number of classes for μ-law encoding

    Returns:
        Argmaxed y_tilde with only ns channels
    """
    assert x.shape[1] == n * μ
    signals = []
    for i in range(n):
        j = i * μ
        signals.append(x[:, j:j + μ, :].argmax(dim=1))
    return torch.cat(signals)
