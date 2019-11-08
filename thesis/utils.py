import random
import torch
import numpy as np
from typing import List


def stereo_to_mono(stereo: torch.Tensor) -> torch.Tensor:
    """
    Convert Stereo to Mono. We do this by simple averaging. This could be catastrophic if L and R are right-shifted but
    that almost never happens.
    :param stereo: the stereo version
    :return: the mono version
    """
    assert stereo.shape[0] == 2
    mono = torch.mean(stereo, dim=0)
    return mono


def set_seed(seed: int):
    """
    Sets the initial random seed for Python, NumPy and PyTorch.
    :param seed: the seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def prime_factorization(n: int) -> List[int]:
    """
    Generates the prime factorization.
    :param n:
    :return: ordered list with all the primes
    """
    i = 2
    factors = []
    while i**2 <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors
