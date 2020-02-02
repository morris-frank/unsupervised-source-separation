from typing import Optional

import torch
from torch import nn

from ...functional import dilate
from ...utils import range_product


class Wavenet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 256,
                 c_channels: Optional[int] = None, n_blocks: int = 3,
                 n_layers: int = 10, residual_width: int = 512,
                 skip_width: int = 256, kernel_size: int = 3):
        super(Wavenet, self).__init__()
        assert kernel_size % 2 != 0
        pad = (kernel_size - 1) // 2

        self.conditional = c_channels is not None
        self.n_blocks, self.n_layers = n_blocks, n_layers
        self.dilations = [2 ** l for _, l in
                          range_product(n_blocks, n_layers)] + [1]

        self.init_conv = nn.Conv1d(in_channels, residual_width, kernel_size,
                                   padding=(kernel_size - 1) // 2)
        self.init_skip = nn.Conv1d(residual_width, skip_width, 1)

        self.filter_conv, self.gate_conv = nn.ModuleList(), nn.ModuleList()
        self.skip_conv, self.feat_conv = nn.ModuleList(), nn.ModuleList()
        self.filter_cond_conv = nn.ModuleList()
        self.gate_cond_conv = nn.ModuleList()
        for _, _ in range_product(self.n_blocks, self.n_layers):
            self.filter_conv.append(
                nn.Conv1d(residual_width, residual_width, kernel_size,
                          padding=pad, bias=not self.conditional))
            self.gate_conv.append(
                nn.Conv1d(residual_width, residual_width, kernel_size,
                          padding=pad, bias=not self.conditional))
            self.skip_conv.append(
                nn.Conv1d(residual_width, skip_width, 1, bias=False))
            self.feat_conv.append(
                nn.Conv1d(residual_width, residual_width, 1, bias=False))

            if self.conditional:
                self.filter_cond_conv.append(
                    nn.Conv1d(c_channels, residual_width, 1, bias=False))
                self.gate_cond_conv.append(
                    nn.Conv1d(c_channels, residual_width, 1, bias=False))

        self.final_skip = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_width, skip_width, 1)
        )

        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_width, out_channels, 1)
        )

    @staticmethod
    def _scale_cond(cond: torch.Tensor, dilation: int, channels: int):
        if cond.ndim == 3:
            cond = dilate(cond, new=dilation, old=1)
        else:
            cond = cond[..., None]
            cond = cond.repeat(channels // cond.shape[0], 1, 1)
        return cond

    def forward(self, x: torch.Tensor,
                conditional: Optional[torch.Tensor]) -> torch.Tensor:
        assert (conditional is not None) == self.conditional

        feat = self.init_conv(x)
        skip = self.init_skip(feat)

        for k in range(self.n_blocks * self.n_layers):
            dilated = dilate(feat, new=self.dilations[k],
                             old=self.dilations[k - 1])

            f = self.filter_conv[k](dilated)
            g = self.gate_conv[k](dilated)

            if self.conditional:
                _f = self.filter_cond_conv[k](conditional)
                _g = self.gate_cond_conv[k](conditional)
                f = f + self._scale_cond(_f, self.dilations[k], f.shape[0])
                g = g + self._scale_cond(_g, self.dilations[k], g.shape[0])

            residual = torch.sigmoid(f) * torch.tanh(g)

            feat = dilated + self.feat_conv[k](residual)
            _skip = self.skip_conv[k](residual)
            skip = skip + dilate(_skip, new=1, old=self.dilations[k])

        skip = self.final_skip(skip)
        out = self.final(skip)
        return out
