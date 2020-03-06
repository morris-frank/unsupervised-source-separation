import math
from typing import Optional as Opt
from functools import partial

import torch
from torch import nn
from torch.nn.utils import weight_norm

from .modules import Conv1d, ZeroConv1d


class GatedResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        cin_channels: Opt[int] = None,
        causal: bool = False,
    ):
        super(GatedResBlock, self).__init__()
        self.causal = causal
        self.cin_channels = cin_channels
        self.conditioned = cin_channels is not None
        self.skip = skip_channels is not None

        self.filter_conv = Conv1d(
            in_channels, out_channels, kernel_size, dilation, causal
        )
        self.gate_conv = Conv1d(
            in_channels, out_channels, kernel_size, dilation, causal
        )
        self.res_conv = weight_norm(nn.Conv1d(out_channels, in_channels, kernel_size=1))
        nn.init.kaiming_normal_(self.res_conv.weight)

        if self.skip:
            self.skip_conv = weight_norm(
                nn.Conv1d(out_channels, skip_channels, kernel_size=1)
            )
            nn.init.kaiming_normal_(self.skip_conv.weight)

        if self.conditioned:
            self.filter_conv_c = weight_norm(
                nn.Conv1d(cin_channels, out_channels, kernel_size=1)
            )
            self.gate_conv_c = weight_norm(
                nn.Conv1d(cin_channels, out_channels, kernel_size=1)
            )
            nn.init.kaiming_normal_(self.filter_conv_c.weight)
            nn.init.kaiming_normal_(self.gate_conv_c.weight)

    def forward(self, tensor, c=None):
        h_filter = self.filter_conv(tensor)
        h_gate = self.gate_conv(tensor)

        if self.conditioned:
            h_filter += self.filter_conv_c(c)
            h_gate += self.gate_conv_c(c)

        out = torch.tanh(h_filter) * torch.sigmoid(h_gate)

        res = self.res_conv(out)
        skip = self.skip_conv(out) if self.skip else None
        return (tensor + res) * math.sqrt(0.5), skip


class Wavenet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        n_blocks: int = 1,
        n_layers: int = 6,
        residual_channels: int = 256,
        gate_channels: int = 256,
        skip_channels: int = 256,
        kernel_size: int = 3,
        cin_channels: Opt[int] = 80,
        causal: bool = False,
        zero_final: bool = False,
    ):
        super(Wavenet, self).__init__()

        self.skip = skip_channels is not None
        self.init = nn.Sequential(
            Conv1d(in_channels, residual_channels, 3, causal=causal), nn.ReLU()
        )

        self.res_blocks = nn.ModuleList()
        for b in range(n_blocks):
            for n in range(n_layers):
                self.res_blocks.append(
                    GatedResBlock(
                        residual_channels,
                        gate_channels,
                        skip_channels,
                        kernel_size,
                        dilation=2 ** n,
                        cin_channels=cin_channels,
                        causal=causal,
                    )
                )

        last_channels = skip_channels if self.skip else residual_channels
        last_layer = ZeroConv1d if zero_final else partial(nn.Conv1d, kernel_size=1)
        self.final = nn.Sequential(
            nn.ReLU(),
            Conv1d(last_channels, last_channels, 1, causal=causal),
            nn.ReLU(),
            last_layer(last_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, c: Opt[torch.Tensor] = None):
        h = self.init(x)
        skip = 0
        for i, block in enumerate(self.res_blocks):
            if self.skip:
                h, s = block(h, c)
                skip += s
            else:
                h, _ = block(h, c)
        if self.skip:
            out = self.final(skip)
        else:
            out = self.final(h)
        return out
