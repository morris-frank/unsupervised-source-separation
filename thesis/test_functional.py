import torch


def test_μ_law():
    from .functional import encode_μ_law, decode_μ_law
    x = torch.tensor([-1, -0.5, 0, 0.5, 0.9], dtype=torch.float)
    y = torch.tensor([-128, -113, 0, 112, 125], dtype=torch.int8)

    assert torch.all(encode_μ_law(x).type(torch.int8) == y)
    assert torch.allclose(decode_μ_law(y), x, atol=0.1)


def test_shift1d():
    from .functional import shift1d
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
