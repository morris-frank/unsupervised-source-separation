import torch
from .utils import prime_factorization, stereo_to_mono


def test_stereo_to_mono():
    stereo = torch.tensor([[2., 4., 8.], [2., 8., 4.]])
    assert torch.allclose(stereo_to_mono(stereo), torch.tensor([2., 6., 6.]))


def test_prime_factorization():
    assert prime_factorization(231) == [3, 7, 11]
