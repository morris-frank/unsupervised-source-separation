from typing import List, Union

import torch
from torch import nn

from .functional import dilate, range_product


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 16,
                 n_blocks: int = 3, n_layers: int = 10, width: int = 128,
                 conditional_dims: List[int] = None, kernel_size: int = 3,
                 pool_size: int = 1, device: str = 'cpu'):
        super(TemporalEncoder, self).__init__()
        assert kernel_size % 2 != 0
        pad = (kernel_size - 1) // 2

        self.n_blocks, self.n_layers = n_blocks, n_layers
        self.dilations = [2 ** l for _, l in
                          range_product(n_blocks, n_layers)] + [1]

        conditional_dims = conditional_dims or []
        self.n_conds = len(conditional_dims)

        self.device = device

        self.init = nn.Conv1d(in_channels, width, kernel_size, padding=pad)
        self.final = nn.Sequential(
            nn.Conv1d(width, out_channels, 1),
            nn.AvgPool1d(pool_size)
        )

        self.residuals_front = nn.ModuleList()
        self.residuals_back = nn.ModuleList()
        self.conditionals = nn.ModuleList(
            nn.ModuleList() for _ in range(self.n_conds))
        for _, _ in range_product(n_blocks, n_layers):
            self.residuals_front.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(width, width, kernel_size, padding=pad, bias=False)
            ))
            self.residuals_back.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(width, width, 1)
            ))
            for ml, dim in zip(self.conditionals, conditional_dims):
                ml.append(nn.Linear(dim, width, bias=False))

    def forward(self, x: torch.Tensor,
                conditionals: Union[torch.Tensor, List[torch.Tensor]] = None) \
            -> torch.Tensor:
        conditionals = conditionals or []
        if isinstance(conditionals, torch.Tensor):
            conditionals = [conditionals]
        assert len(conditionals) == self.n_conds

        y = self.init(x)
        for i in range(self.n_blocks * self.n_layers):
            # Increase dilation by one step
            y = dilate(y, new=self.dilations[i], old=self.dilations[i - 1])
            _y = self.residuals_front[i](y)

            for j in range(self.n_conds):
                c = self.conditionals[j][i](conditionals[j])[..., None]
                # As we compound dilate the y we have to repeat the conditional
                # along the batch dimension
                c = c.repeat(y.shape[0] // c.shape[0], 1, 1)
                _y = _y + c

            y = y + self.residuals_back[i](_y)
        # Remove the compound dilations
        y = dilate(y, new=self.dilations[-1], old=self.dilations[-2])
        y = self.final(y)
        return y
