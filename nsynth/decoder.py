from typing import List, Union

import torch
from torch import nn

from .functional import range_product, dilate


class WavenetDecoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 256,
                 n_blocks: int = 3, n_layers: int = 10,
                 residual_width: int = 512, skip_width: int = 256,
                 conditional_dims: List[int] = None, kernel_size: int = 3):
        super(WavenetDecoder, self).__init__()
        if conditional_dims is None:
            conditional_dims = [(16, False)]

        assert kernel_size % 2 != 0
        self.n_conds = len(conditional_dims)
        self.n_blocks, self.n_layers = n_blocks, n_layers

        # TODO: prime factors
        self.dilations = [2 ** l for _, l in
                          range_product(n_blocks, n_layers)] + [1]

        self.init_conv = nn.Conv1d(in_channels, residual_width, kernel_size,
                                   padding=(kernel_size - 1) // 2)
        self.init_skip = nn.Conv1d(residual_width, skip_width, 1)

        self.filter_conv = self._make_module_list(
            residual_width, residual_width, kernel_size)
        self.gate_conv = self._make_module_list(
            residual_width, residual_width, kernel_size)
        self.skip_conv = self._make_module_list(
            residual_width, skip_width, 1)
        self.feat_conv = self._make_module_list(
            residual_width, residual_width, 1)

        self.filter_cond_conv = nn.ModuleList()
        self.gate_cond_conv = nn.ModuleList()
        for dim, lin in conditional_dims:
            self.filter_cond_conv.append(
                self._make_module_list(dim, residual_width, 1, lin)
            )
            self.gate_cond_conv.append(
                self._make_module_list(dim, residual_width, 1, lin)
            )

        self.final_skip = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_width, skip_width, 1)
        )

        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_width, out_channels, 1)
        )

    def _make_module_list(self, in_channels: int, out_channels: int,
                          kernel_size: int, linear: bool = False) \
            -> nn.ModuleList:
        module_list = []
        pad = (kernel_size - 1) // 2
        for _, layer in range_product(self.n_blocks, self.n_layers):
            if linear:
                ly = nn.Linear(in_channels, out_channels)
            else:
                ly = nn.Conv1d(in_channels, out_channels, kernel_size,
                               bias=False, padding=pad)
            module_list.append(ly)
        return nn.ModuleList(module_list)

    @staticmethod
    def _scale_cond(cond: torch.Tensor, dilation: int, channels: int):
        if cond.ndim == 3:
            cond = dilate(cond, new=dilation, old=1)
        else:
            cond = cond[..., None]
            cond = cond.repeat(channels // cond.shape[0], 1, 1)
        return cond

    def forward(self, x: torch.Tensor,
                conditionals: Union[torch.Tensor, List[torch.Tensor]]) \
            -> torch.Tensor:
        if isinstance(conditionals, torch.Tensor):
            conditionals = [conditionals]
        assert len(conditionals) == self.n_conds
        feat = self.init_conv(x)
        skip = self.init_skip(feat)

        for i in range(self.n_blocks * self.n_layers):
            dilated = dilate(feat, new=self.dilations[i],
                             old=self.dilations[i - 1])

            f = self.filter_conv[i](dilated)
            g = self.gate_conv[i](dilated)

            # Now add all the conditionals to the filters and gates
            for j in range(self.n_conds):
                _f = self.filter_cond_conv[j][i](conditionals[j])
                _g = self.gate_cond_conv[j][i](conditionals[j])
                f = f + self._scale_cond(_f, self.dilations[i], f.shape[0])
                g = g + self._scale_cond(_g, self.dilations[i], g.shape[0])

            residual = torch.sigmoid(f) * torch.tanh(g)

            feat = dilated + self.feat_conv[i](residual)
            _skip = self.skip_conv[i](residual)
            skip = skip + dilate(_skip, new=1, old=self.dilations[i])

        skip = self.final_skip(skip)
        out = self.final(skip)
        return out
