from itertools import product

import torch
from torch import nn

from .modules import BlockWiseConv1d


def test_blockwiseconv1d():
    """
    Right now just testing that 1×1 convs are the same as the normal torch Conv1d
    """
    n_batch, n_channels, length = 8, 2, 32
    n_filters = 5
    x = torch.rand((n_batch, n_channels, length))
    conv = nn.Conv1d(in_channels=n_channels, out_channels=n_filters, kernel_size=1)
    for block_size, causal in product([1, 4], [True, False]):
        block_conv = BlockWiseConv1d(in_channels=n_channels, out_channels=n_filters, block_size=block_size,
                                     kernel_size=1, causal=causal)
        block_conv.weight, block_conv.bias = conv.weight, conv.bias
        assert torch.all(conv(x) == block_conv(x))
