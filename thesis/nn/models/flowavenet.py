from math import log
from math import tau as τ

import torch
from torch import nn
from torch.nn import functional as F

from . import BaseModel
from ..modules import STFTUpsample
from ..wavenet import Wavenet
from ...dist import norm_log_prob
from ...functional import (
    permute_L2C,
    flip,
    permute_C2L,
    chunk,
    interleave,
)
from ...utils import clean_init_args
from ...setup import DEFAULT


class ActNorm(nn.Module):
    def __init__(self, in_channel, pretrained=False):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1))
        self.initialized = pretrained

    def initialize(self, x):
        with torch.no_grad():
            flatten = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, x):
        if not self.initialized:
            if self.training:
                self.initialize(x)
            self.initialized = True

        B, _, T = x.size()
        log_abs = self.scale.abs().log()
        log_det = torch.sum(log_abs) * B * T

        return self.scale * (x + self.loc), log_det

    def reverse(self, y):
        return y / self.scale - self.loc


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, width=256, num_layer=6, groups=1, cin_channel=None):
        super().__init__()

        self.groups = groups
        if cin_channel is not None:
            cin_channel = cin_channel // 2 * groups

        self.net = Wavenet(
            in_channels=in_channel // 2 * groups,
            out_channels=in_channel * groups,
            n_blocks=1,
            n_layers=num_layer,
            residual_channels=width * groups,
            gate_channels=width * groups,
            skip_channels=width * groups,
            kernel_size=3,
            cin_channels=cin_channel,
            groups=groups,
            causal=False,
            zero_final=True,
            bias=False,
        )

    def forward(self, x, c=None):
        in_a, in_b = chunk(x, groups=self.groups)

        if c is not None:
            c, _ = chunk(c, groups=self.groups)

        log_s, t = chunk(self.net(in_a, c), groups=self.groups)

        out_b = (in_b - t) * torch.exp(-log_s)
        log_det = torch.sum(-log_s)

        return interleave((in_a, out_b), groups=self.groups), log_det

    def reverse(self, y, c=None):
        out_a, out_b = chunk(y, groups=self.groups)

        if c is not None:
            c, _ = chunk(c, groups=self.groups)

        log_s, t = chunk(self.net(out_a, c), groups=self.groups)
        in_b = out_b * torch.exp(log_s) + t

        return interleave((out_a, in_b), groups=self.groups)


class Flow(nn.Module):
    def __init__(
        self, in_channel, width, num_layer, groups=1, cin_channel=None,
    ):
        super().__init__()
        self.groups = groups

        self.actnorm = ActNorm(in_channel * groups)
        self.coupling = AffineCoupling(
            in_channel,
            width=width,
            num_layer=num_layer,
            groups=groups,
            cin_channel=cin_channel,
        )

    def forward(self, x, c=None):
        out, log_det = self.actnorm(x)
        out, log_det_c = self.coupling(out, c)

        out = flip(out, groups=self.groups)
        if c is not None:
            c = flip(c, groups=self.groups)

        if log_det_c is not None:
            log_det += log_det_c

        return out, c, log_det

    def reverse(self, out, c=None):
        out = flip(out, groups=self.groups)

        if c is not None:
            c = flip(c, groups=self.groups)

        x = self.coupling.reverse(out, c)
        x = self.actnorm.reverse(x)
        return x, c


class Block(nn.Module):
    def __init__(
        self,
        in_channel,
        n_flow,
        n_layer,
        width,
        split=False,
        cin_channel=None,
        groups=1,
    ):
        super().__init__()

        self.groups = groups
        self.split = split
        squeeze_dim = in_channel * 2
        if cin_channel is not None:
            cin_channel = cin_channel * 2

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(
                Flow(
                    squeeze_dim,
                    width=width,
                    num_layer=n_layer,
                    cin_channel=cin_channel,
                    groups=groups,
                )
            )

        if cin_channel is not None:
            cin_channel *= groups

        if self.split:
            self.prior = Wavenet(
                in_channels=squeeze_dim // 2 * groups,
                out_channels=squeeze_dim * groups,
                n_blocks=1,
                n_layers=2,
                residual_channels=width * groups,
                gate_channels=width * groups,
                skip_channels=width * groups,
                kernel_size=3,
                cin_channels=cin_channel,
                causal=False,
                zero_final=True,
                bias=False,
                groups=groups,
            )

    def forward(self, x, c=None):
        x = permute_L2C(x)

        if c is not None:
            c = permute_L2C(c)

        log_det = 0
        for flow in self.flows:
            x, c, _log_det = flow(x, c)
            log_det = log_det + _log_det

        log_p, z = None, None
        if self.split:
            x, z = chunk(x, groups=self.groups)
            μ, σ = chunk(self.prior(x, c), groups=self.groups)
            N, _, L = μ.shape
            log_p = norm_log_prob(z, μ, σ)

        return x, c, log_det, log_p, z

    def reverse(self, y, c=None, eps=None):
        if self.split:
            μ, σ = chunk(self.prior(y, c), groups=self.groups)
            z = μ + σ.exp() * eps
            x = interleave((y, z), groups=self.groups)
        else:
            x = y

        for flow in self.flows[::-1]:
            x, c = flow.reverse(x, c)

        unsqueezed_x = permute_C2L(x)
        if c is not None:
            c = permute_C2L(c)
        return unsqueezed_x, c


class Flowavenet(BaseModel):
    def __init__(
        self,
        in_channel,
        n_block,
        n_flow,
        n_layer,
        width,
        block_per_split,
        cin_channel=None,
        groups=1,
        **kwargs,
    ):
        super(Flowavenet, self).__init__(**kwargs)
        self.params = clean_init_args(locals().copy())
        self.groups = groups
        self.block_per_split, self.n_block = block_per_split, n_block

        if cin_channel is not None:
            self.c_up = STFTUpsample([16, 16])

        self.blocks = nn.ModuleList()
        for i in range(self.n_block):
            split = (i < self.n_block - 1) and (i + 1) % self.block_per_split == 0

            self.blocks.append(
                Block(
                    in_channel,
                    n_flow,
                    n_layer,
                    split=split,
                    width=width,
                    cin_channel=cin_channel,
                    groups=groups,
                )
            )
            if cin_channel is not None:
                cin_channel *= 2
            if not split:
                in_channel *= 2

    def forward(self, x, c=None):
        N, C, L = x.size()
        out = x
        if c is not None:
            c = self.c_up(c, L)

        log_det = 0
        log_p_list, z_list = [], []
        for i, block in enumerate(self.blocks):
            out, c, log_det_new, log_p_new, z_new = block(out, c)
            log_det = log_det + log_det_new
            if z_new is not None:
                z_list.append(z_new)
                log_p_list.append(log_p_new)

        z_list.append(out)
        log_p_out = -.5 * (log(τ) + out.pow(2))
        log_p_list.append(log_p_out)
        
        for i in range(len(log_p_list)):
            _log_p = log_p_list[i].mean((0, -1)).view(self.groups, -1).mean(-1)
            for k in range(self.groups):
                setattr(self.ℒ, f"log_p_{i}/{DEFAULT.signals[k]}", _log_p[k])

        log_p = self.combine_z_list(log_p_list)
        z = self.combine_z_list(z_list)

        log_det = log_det / (N * C * L)

        return z, log_p, log_det

    def combine_z_list(self, z_list):
        for i in reversed(range(self.n_block)):
            if not ((i + 1) % self.block_per_split or i == self.n_block - 1):
                z1 = z_list.pop()
                z2 = z_list.pop()
                z_list.append(interleave((z1, z2), groups=self.groups))
            z_list[-1] = permute_C2L(z_list[-1])
        return z_list[0]

    def reverse(self, z, c=None):
        if c is not None:
            L, LC = z.shape[-1], c.shape[-1]
            if L != LC:
                c = self.c_up(c, L)

        z_list = []
        x = z
        for i in range(self.n_block):
            x = permute_L2C(x)
            if c is not None:
                c = permute_L2C(c)
            if not ((i + 1) % self.block_per_split or i == self.n_block - 1):
                x, z = chunk(x, groups=self.groups)
                z_list.append(z)

        for i, block in enumerate(self.blocks[::-1]):
            index = self.n_block - i
            # print(f"bw {index}: {x.mean()}")
            if not (index % self.block_per_split or index == self.n_block):
                x, c = block.reverse(x, c, z_list[index // self.block_per_split - 1])
                # print('now')
            else:
                x, c = block.reverse(x, c)
        return x

    def test(self, x):
        N = x.shape[1]
        if x.dim() > 3:
            x = x.flatten(1, 2)
        _, log_p, log_det = self.forward(x)

        self.ℒ.log_det = -torch.mean(log_det)
        ℒ = self.ℒ.log_det

        log_p = -log_p.mean((0, -1))
        for k in range(N):
            ℒ += log_p[k]
            setattr(self.ℒ, f"log_p/{DEFAULT.signals[k]}", log_p[k])
        return ℒ


class FlowavenetClassified(Flowavenet):
    def __init__(self, *args, **kwargs):
        super(FlowavenetClassified, self).__init__(*args, **kwargs)
        self.classifier = Wavenet(
            in_channels=2560,
            out_channels=4,
            residual_channels=32,
            skip_channels=None,
            gate_channels=32,
            cin_channels=None,
        )

    def forward(self, x, c=None):
        out = log_p, log_det = super(FlowavenetClassified, self).forward(x, c)
        ŷ = self.classifier(out)
        ŷ = torch.sigmoid(ŷ).squeeze().mean(-1)
        return ŷ, log_p, log_det

    def test(self, m, y):
        ŷ, log_p, log_det = self.forward(m)

        self.ℒ.ce = F.cross_entropy(ŷ, y)
        self.ℒ.log_det = -torch.mean(log_det)
        self.ℒ.log_p = -log_p.mean()

        ℒ = self.ℒ.log_p + self.ℒ.log_det + self.ℒ.ce
        return ℒ
