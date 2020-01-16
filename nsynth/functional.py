import math
import torch
import torch.nn.functional as F
from itertools import product


def range_product(*args: int):
    """
    Gives an iterator over the product of the ranges of the given integers.
    Args:
        *args: A number of Integers

    Returns:

    """
    return product(*map(range, args))


def dilate(x: torch.Tensor, new: int, old: int = 1) -> torch.Tensor:
    """
    :param x: The input Tensor
    :param new: The new dilation we want
    :param old: The dilation x already has
    """
    [N, C, L] = x.shape  # N == Batch size × old
    dilation = new / old
    if dilation == 1:
        return x
    L, N = int(L / dilation), int(N * dilation)
    x = x.permute(1, 2, 0)
    x = torch.reshape(x, [C, L, N])
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


def encode_μ_law(x: torch.Tensor, μ: int = 255, cast: bool = False)\
        -> torch.Tensor:
    """
    Encodes the input tensor element-wise with μ-law encoding

    Args:
        x: tensor
        μ: the size of the encoding (number of possible classes)
        cast: whether to cast to int8

    Returns:

    """
    out = torch.sign(x) * torch.log(1 + μ * torch.abs(x)) / math.log(1 + μ)
    out = torch.floor(out * math.ceil(μ / 2))
    if cast:
        out = out.type(torch.int8)
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
