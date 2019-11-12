import numpy as np
import torch


def dilate(x: torch.Tensor, target_dilation: int) -> torch.Tensor:
    prev_dilation, channels, length = x.size()  # n: prev dilation, c: num of channels, l: input length

    if prev_dilation == target_dilation:
        # Already have target size, nothing to dilate
        return x

    dilation_factor = target_dilation / prev_dilation

    new_l = int(np.ceil(length / dilation_factor) * dilation_factor)
    if new_l != length:
        x = pad1d(x, new_l, dim=2)

    # reshape according to dilation
    x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    x = x.view(channels, new_l, target_dilation)
    x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return x


def pad1d(x: torch.Tensor, new_l: int, dim: int) -> torch.Tensor:
    # TODO
    return x


def time_to_batch(x: torch.Tensor, block_size: int) -> torch.Tensor:
    assert x.ndim == 3
    batch_size, channels, length = x.shape

    y = torch.reshape(x, [batch_size, channels, length // block_size, block_size])
    y = y.permute(0, 3, 1, 2)
    y = torch.reshape(y, [batch_size * block_size, channels, length // block_size])
    return y


def batch_to_time(x: torch.Tensor, block_size: int) -> torch.Tensor:
    assert x.ndim == 3
    batch_size, channels, k = x.shape
    y = torch.reshape(x, [batch_size // block_size, block_size, channels, k])
    y = y.permute(0, 2, 3, 1)
    y = torch.reshape(y, [batch_size // block_size, channels, k * block_size])
    return y
