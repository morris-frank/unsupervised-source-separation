import math
import random
from typing import List

import numpy as np
import torch


def stereo_to_mono(stereo: torch.Tensor) -> torch.Tensor:
    """
    Convert Stereo to Mono. We do this by simple averaging. This could be catastrophic if L and R are right-shifted but
    that almost never happens

    Args:
        stereo:  the stereo version

    Returns:
        the mono version
    """
    assert stereo.shape[0] == 2
    mono = torch.mean(stereo, dim=0)
    return mono


def set_seed(seed: int):
    """
    Sets the initial random seed for Python, NumPy and PyTorch.

    Args:
        seed: the seed

    Returns:

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def prime_factorization(n: int) -> List[int]:
    """
    Generates the prime factorization.

    Args:
        n:

    Returns:
        ordered list with all the primes
    """
    i = 2
    factors = []
    while i ** 2 <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def encode_μ_law(x: torch.Tensor, μ: int = 255, cast: bool = False) -> torch.Tensor:
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
