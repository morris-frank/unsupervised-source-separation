import random

import numpy as np
import torch

from .utils import prime_factorization, stereo_to_mono, set_seed, encode_μ_law, decode_μ_law


def test_stereo_to_mono():
    stereo = torch.tensor([[2., 4., 8.], [2., 8., 4.]])
    mono = stereo_to_mono(stereo)
    assert torch.allclose(mono, torch.tensor([2., 6., 6.]))
    assert mono.is_contiguous()


def test_prime_factorization():
    assert prime_factorization(231) == [3, 7, 11]


def test_set_sed():
    set_seed(42)
    assert np.random.randint(0, 100) == 51
    assert random.randint(0, 100) == 81
    assert torch.randint(100, (1,)).item() == 42


def test_μ_law():
    x = torch.tensor([-1, -0.5, 0, 0.5, 0.9], dtype=torch.float)
    y = torch.tensor([-128, -113, 0, 112, 125], dtype=torch.int8)

    assert torch.all(encode_μ_law(x, cast=True) == y)
    assert torch.allclose(decode_μ_law(y), x, atol=0.1)
