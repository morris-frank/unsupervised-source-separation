import torch

from .functional import time_to_batch, batch_to_time


def test_time_to_batch():
    dilation = 8
    nbatch, nchannel, length = 8, 2, 32
    x = torch.rand((nbatch, nchannel, length))
    ttb = time_to_batch(x, dilation)
    assert list(ttb.shape) == [nbatch * dilation, nchannel, length // dilation]

    _ttb = batch_to_time(ttb, dilation)
    assert x.shape == _ttb.shape
    assert torch.all(x == _ttb)
