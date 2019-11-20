import torch

from .functional import time_to_batch, batch_to_time, shift1d


def test_time_to_batch():
    n_batch, n_channel, length = 8, 2, 32
    dilation = 8
    x = torch.rand((n_batch, n_channel, length))
    ttb = time_to_batch(x, dilation)
    assert list(ttb.shape) == [n_batch * dilation, n_channel, length // dilation]

    _ttb = batch_to_time(ttb, dilation)
    assert x.shape == _ttb.shape
    # B2T is the inverse of T2B so we should get out the same thing again:
    assert torch.all(x == _ttb)
    assert ttb.is_contiguous()
    assert _ttb.is_contiguous()


def test_shift1d():
    n_batch, n_channel, length = 8, 2, 32
    x = torch.rand((n_batch, n_channel, length))
    for shift in [1, 3, 5]:
        y = shift1d(x, -shift)
        assert y.shape == x.shape
        assert torch.all(y[:, :, shift:] == x[:, :, :length - shift])
        assert y.is_contiguous()
        y = shift1d(x, shift)
        assert y.shape == x.shape
        assert torch.all(y[:, :, :length - shift] == x[:, :, shift:])
        assert y.is_contiguous()
