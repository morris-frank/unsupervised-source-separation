import torch
from torch import nn

from .functional import time_to_batch, batch_to_time


class BlockWiseConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 block_size: int,
                 causal: bool = False,
                 **kwargs):
        super(BlockWiseConv1d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.block_size = block_size

        assert kernel_size % 2 != 0
        assert block_size >= 1

        if causal:
            pad = (kernel_size - 1, 0)
        else:
            pad = ((kernel_size - 1) // 2, (kernel_size - 1) // 2)
        self.constant_pad = nn.ConstantPad1d(pad, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = time_to_batch(x, self.block_size)
        y = self.constant_pad(y)
        y = super(BlockWiseConv1d, self).forward(y)
        y = batch_to_time(y, self.block_size)
        return y
