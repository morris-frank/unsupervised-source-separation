from typing import Optional

import torch
from torch import nn
from torch.nn.utils import weight_norm

from ...functional import dilate, remove_list_weight_norm
from ...utils import range_product


class Wavenet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 256,
                 c_channels: Optional[int] = None, n_blocks: int = 3,
                 n_layers: int = 10, residual_width: int = 512,
                 skip_width: int = 256, kernel_size: int = 3,
                 normalize: bool = True):
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

        self.skip, self.thru = nn.ModuleList(), nn.ModuleList()
        self.gate, self.feat = nn.ModuleList(), nn.ModuleList()
        self.gate_c, self.feat_c = nn.ModuleList(), nn.ModuleList()
        for _, _ in range_product(self.n_blocks, self.n_layers):
            self.gate.append(
                nn.Conv1d(residual_width, residual_width, kernel_size,
                          padding=pad, bias=not self.conditional))
            self.feat.append(
                nn.Conv1d(residual_width, residual_width, kernel_size,
                          padding=pad, bias=not self.conditional))
            self.skip.append(
                nn.Conv1d(residual_width, skip_width, 1, bias=False))
            self.thru.append(
                nn.Conv1d(residual_width, residual_width, 1, bias=False))

            if normalize:
                self.gate[-1] = weight_norm(self.gate[-1], name='weight')
                self.feat[-1] = weight_norm(self.feat[-1], name='weight')
                self.skip[-1] = weight_norm(self.skip[-1], name='weight')
                self.thru[-1] = weight_norm(self.thru[-1], name='weight')

            if self.conditional:
                self.gate_c.append(
                    nn.Conv1d(c_channels, residual_width, 1, bias=False))
                self.feat_c.append(
                    nn.Conv1d(c_channels, residual_width, 1, bias=False))

                if normalize:
                    self.gate_c[-1] = weight_norm(self.gate_c[-1],
                                                  name='weight')
                    self.feat_c[-1] = weight_norm(self.feat_c[-1],
                                                  name='weight')

        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_width, skip_width, 1),
            nn.ReLU(),
            nn.Conv1d(skip_width, out_channels, 1)
        )

    def forward(self, x: torch.Tensor,
                conditional: Optional[torch.Tensor]) -> torch.Tensor:
        assert (conditional is not None) == self.conditional

        x = self.init_conv(x)
        s = self.init_skip(x)

        for k in range(self.n_blocks * self.n_layers):
            dilated = dilate(x, new=self.dilations[k],
                             old=self.dilations[k - 1])
            g, f = self.gate[k](dilated), self.feat[k](dilated)

            if self.conditional:
                dilated_c = dilate(conditional, self.dilations[k], 1)
                g = g + self.gate_c[k](dilated_c)
                f = f + self.feat_c[k](dilated_c)

            res = torch.sigmoid(g) * torch.tanh(f)

            x = dilated + self.thru[k](res)
            s = s + dilate(self.skip[k](res), 1, self.dilations[k])

        return self.final(s)

    def train(self, mode: bool = True):
        if mode is False:
            self.remove_weight_norm()
        return super(Wavenet, self).train()

    def remove_weight_norm(self):
        """
        Removes all the weight norms for this Wavenet
        """
        self.gate = remove_list_weight_norm(self.gate)
        self.feat = remove_list_weight_norm(self.feat)
        self.thru = remove_list_weight_norm(self.thru)
        self.skip = remove_list_weight_norm(self.skip)
        self.gate_c = remove_list_weight_norm(self.gate_c)
        self.feat_c = remove_list_weight_norm(self.feat_c)
