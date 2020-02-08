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
                 normalize: bool = False):
        super(Wavenet, self).__init__()
        assert kernel_size % 2 != 0
        pad = (kernel_size - 1) // 2

        self.normalized = normalize
        wn = weight_norm if normalize else lambda x: x
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
            self.gate.append(wn(
                nn.Conv1d(residual_width, residual_width, kernel_size,
                          padding=pad, bias=not self.conditional)))
            self.feat.append(wn(
                nn.Conv1d(residual_width, residual_width, kernel_size,
                          padding=pad, bias=not self.conditional)))
            self.skip.append(wn(
                nn.Conv1d(residual_width, skip_width, 1, bias=False)))
            self.thru.append(wn(
                nn.Conv1d(residual_width, residual_width, 1, bias=False)))

            if self.conditional:
                self.gate_c.append(wn(
                    nn.Conv1d(c_channels, residual_width, 1, bias=False)))
                self.feat_c.append(wn(
                    nn.Conv1d(c_channels, residual_width, 1, bias=False)))

        self.final = nn.Conv1d(skip_width, out_channels, 1)
        #final.weight.data.zero_()
        #final.bias.data.zero_()
        #self.final = nn.Sequential(final, nn.Tanh())

    def forward(self, x: torch.Tensor,
                conditional: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert (conditional is not None) == self.conditional

        h = self.init_conv(x)
        s = self.init_skip(h)

        for k in range(self.n_blocks * self.n_layers):
            dilated = dilate(h, new=self.dilations[k],
                             old=self.dilations[k - 1])
            g, f = self.gate[k](dilated), self.feat[k](dilated)

            if self.conditional:
                dilated_c = dilate(conditional, self.dilations[k], 1)
                g = g + self.gate_c[k](dilated_c)
                f = f + self.feat_c[k](dilated_c)

            res = torch.sigmoid(g) * torch.tanh(f)

            h = dilated + self.thru[k](res)
            s = s + dilate(self.skip[k](res), 1, self.dilations[k])
        return self.final(s)

    def train(self, mode: bool = True):
        if mode is False and self.normalized:
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
