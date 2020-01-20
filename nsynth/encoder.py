from typing import List, Union

import torch
from torch import nn

from .functional import dilate, range_product


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 16,
                 n_blocks: int = 3, n_layers: int = 10, width: int = 128,
                 kernel_size: int = 3, pool_size: int = 512):
        """

        Args:
            in_channels: Number of input in_channels
            out_channels: Number of output in_channels
            n_blocks: Number of dilation blocks / stages
            n_layers: Number of layers in each stage / block
            width: Width of the hidden layers
            kernel_size: General Kernel size for the convs
            pool_size: Size of avg pooling before the output
        """
        super(TemporalEncoder, self).__init__()

        assert kernel_size % 2 != 0
        pad = (kernel_size - 1) // 2

        # Go from input in_channels to hidden width:
        self.init = nn.Conv1d(in_channels, width, kernel_size, padding=pad)
        # Go from hidden width to final output in_channels
        self.final = nn.Sequential(
            nn.Conv1d(width, out_channels, 1),
            nn.AvgPool1d(kernel_size=pool_size)
        )

        self.residuals = []
        for _, layer in range_product(n_blocks, n_layers):
            self.residuals.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(width, width, kernel_size, padding=pad),
                nn.ReLU(),
                nn.Conv1d(width, width, 1)
            ))
        self.residuals = nn.ModuleList(self.residuals)

        # TODO: replace with prime factors
        self.dilations = [2 ** l for _, l in range_product(n_blocks, n_layers)]
        self.dilations.append(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.init(x)
        for i, residual in enumerate(self.residuals):
            # Increase dilation by one step
            y = dilate(y, new=self.dilations[i], old=self.dilations[i - 1])
            y = y + residual(y)
        # Remove the compound dilations
        y = dilate(y, new=self.dilations[-1], old=self.dilations[-2])
        y = self.final(y)
        return y


class ConditionalTemporalEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 16,
                 n_blocks: int = 3, n_layers: int = 10, width: int = 128,
                 conditional_dims: List[int] = None, kernel_size: int = 3,
                 pool_size: int = 1, device: str = 'cpu'):
        super(ConditionalTemporalEncoder, self).__init__()

        assert kernel_size % 2 != 0
        pad = (kernel_size - 1) // 2

        self.n_conds = len(conditional_dims)

        self.device = device

        self.init = nn.Conv1d(in_channels, width, kernel_size, padding=pad)
        self.final = nn.Sequential(
            nn.Conv1d(width, out_channels, 1),
            nn.AvgPool1d(pool_size)
        )

        self.residuals_front = nn.ModuleList()
        self.residuals_back = nn.ModuleList()
        if conditional_dims:
            self.conditionals = nn.ModuleList(
                nn.ModuleList() for _ in range(self.n_conds))
        for _, layer in range_product(n_blocks, n_layers):
            self.residuals_front.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(width, width, kernel_size, padding=pad, bias=False)
            ))
            for lay, dim in zip(self.conditionals, conditional_dims):
                lay.append(nn.Linear(dim, width, bias=False))
            self.residuals_back.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(width, width, 1)
            ))

        # TODO: replace with prime factors
        self.dilations = [2 ** l for _, l in
                          range_product(n_blocks, n_layers)] + [1]

    def forward(self, x: torch.Tensor,
                conditionals: Union[torch.Tensor, List[torch.Tensor]]) \
            -> torch.Tensor:
        if isinstance(conditionals, torch.Tensor):
            conditionals = [conditionals]
        assert len(conditionals) == self.n_conds
        y = self.init(x)
        for i, (front, cond, back) in enumerate(
                zip(self.residuals_front, self.conditionals,
                    self.residuals_back)):
            # Increase dilation by one step
            y = dilate(y, new=self.dilations[i], old=self.dilations[i - 1])
            _y = front(y)

            for i in range(self.n_conds):
                c = cond[i](conditionals[i])[..., None]
                # As we compound dilate the y we have to repeat the conditional
                # along the batch dimension
                c = c.repeat(y.shape[0] // c.shape[0], 1, 1)
                _y = _y + c

            y = y + back(_y)
        # Remove the compound dilations
        y = dilate(y, new=self.dilations[-1], old=self.dilations[-2])
        y = self.final(y)
        return y
