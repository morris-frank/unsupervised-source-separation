from math import log, pi

import torch
from torch import nn
from torch.nn import functional as F

from . import BaseModel
from ..modules import STFTUpsample
from ..wavenet import Wavenet
from ...dist import likelihood_normal
from ...functional import (
    split_LtoC,
    flip_channels,
    split_CtoL,
    chunk_grouped,
    interleave,
)
from ...utils import clean_init_args


class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1), requires_grad=True)
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1), requires_grad=True)

    def forward(self, x):
        B, _, T = x.size()

        log_abs = self.scale.abs().log()

        logdet = torch.sum(log_abs) * B * T

        return self.scale * (x + self.loc), logdet

    def reverse(self, output):
        return output / self.scale - self.loc


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
        in_a, in_b = chunk_grouped(x, self.groups)

        if c is not None:
            c, _ = chunk_grouped(c, self.groups)

        log_s, t = chunk_grouped(self.net(in_a, c), self.groups)

        out_b = (in_b - t) * torch.exp(-log_s)
        logdet = torch.sum(-log_s)

        return interleave((in_a, out_b), self.groups), logdet

    def reverse(self, output, c=None):
        out_a, out_b = chunk_grouped(output, self.groups)

        if c is not None:
            c, _ = chunk_grouped(c, self.groups)

        log_s, t = chunk_grouped(self.net(out_a, c), self.groups)
        in_b = out_b * torch.exp(log_s) + t

        return interleave((out_a, in_b), self.groups)


class Flow(nn.Module):
    def __init__(
        self, in_channel, width, num_layer, groups=1, cin_channel=None,
    ):
        super().__init__()

        self.actnorm = ActNorm(in_channel)
        self.coupling = AffineCoupling(
            in_channel,
            width=width,
            num_layer=num_layer,
            groups=groups,
            cin_channel=cin_channel,
        )

    def forward(self, x, c=None):
        out, logdet = self.actnorm(x)
        out, det = self.coupling(out, c)
        out = flip_channels(out)

        if c is not None:
            c = flip_channels(c)

        if det is not None:
            logdet = logdet + det

        return out, c, logdet

    def reverse(self, out, c=None):
        out = flip_channels(out)

        if c is not None:
            c = flip_channels(c)

        x = self.coupling.reverse(out, c)
        x = self.actnorm.reverse(x)
        return x, c


class Block(nn.Module):
    def __init__(
        self, in_channel, n_flow, n_layer, width, split=False, cin_channel=None,
    ):
        super().__init__()

        self.split = split
        squeeze_dim = in_channel * 2
        if cin_channel is not None:
            cin_channel = cin_channel * 2

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(
                Flow(
                    squeeze_dim, width=width, num_layer=n_layer, cin_channel=cin_channel
                )
            )

        if self.split:
            self.prior = Wavenet(
                in_channels=squeeze_dim // 2,
                out_channels=squeeze_dim,
                n_blocks=1,
                n_layers=2,
                residual_channels=width,
                gate_channels=width,
                skip_channels=width,
                kernel_size=3,
                cin_channels=cin_channel,
                causal=False,
                zero_final=True,
                bias=False,
            )

    def forward(self, x, c=None):
        x = split_LtoC(x)

        if c is not None:
            c = split_LtoC(c)

        logdet, log_p = 0, 0
        for k, flow in enumerate(self.flows):
            x, c, det = flow(x, c)
            logdet = logdet + det

        if self.split:
            x, z = x.chunk(2, 1)
            # WaveNet prior
            mean, log_sd = self.prior(x, c).chunk(2, 1)
            log_p = likelihood_normal(z, mean, log_sd).sum(1, keepdim=True)

        return x, c, logdet, log_p

    def reverse(self, output, c, eps=None):
        if self.split:
            mean, log_sd = self.prior(output, c).chunk(2, 1)
            z_new = mean + log_sd.exp() * eps

            x = torch.cat([output, z_new], 1)
        else:
            x = output

        for i, flow in enumerate(self.flows[::-1]):
            x, c = flow.reverse(x, c)

        unsqueezed_x = split_CtoL(x)
        unsqueezed_c = split_CtoL(c)

        return unsqueezed_x, unsqueezed_c


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
        **kwargs,
    ):
        super(Flowavenet, self).__init__(**kwargs)
        self.params = clean_init_args(locals().copy())
        self.block_per_split, self.n_block = block_per_split, n_block

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
                )
            )
            if cin_channel is not None:
                cin_channel *= 2
            if not split:
                in_channel *= 2

    def forward(self, x, c=None):
        B, _, T = x.size()
        logdet, log_p_sum = 0, 0
        out = x

        if c is not None:
            c = self.c_up(c, T)

        for k, block in enumerate(self.blocks):
            out, c, logdet_new, logp_new = block(out, c)
            logdet = logdet + logdet_new
            if isinstance(logp_new, torch.Tensor):
                logp_new = F.interpolate(logp_new, size=T)
            log_p_sum = logp_new + log_p_sum

        out = F.interpolate(out, size=T)
        log_p_sum += -0.5 * (log(2.0 * pi) + out.pow(2)).sum(1, keepdim=True)
        logdet = logdet / (B * T)
        return log_p_sum, logdet

    def reverse(self, z, c=None):
        _, _, T = z.size()
        _, _, t_c = c.size()
        if T != t_c:
            c = self.c_up(c, T)
        z_list = []
        x = z
        for i in range(self.n_block):
            b_size, _, T = x.size()
            squeezed_x = x.view(b_size, -1, T // 2, 2).permute(0, 1, 3, 2)
            x = squeezed_x.contiguous().view(b_size, -1, T // 2)
            squeezed_c = c.view(b_size, -1, T // 2, 2).permute(0, 1, 3, 2)
            c = squeezed_c.contiguous().view(b_size, -1, T // 2)
            if not ((i + 1) % self.block_per_split or i == self.n_block - 1):
                x, z = x.chunk(2, 1)
                z_list.append(z)

        for i, block in enumerate(self.blocks[::-1]):
            index = self.n_block - i
            if not (index % self.block_per_split or index == self.n_block):
                x, c = block.reverse(x, c, z_list[index // self.block_per_split - 1])
            else:
                x, c = block.reverse(x, c)
        return x

    def test(self, s, m):
        m = F.interpolate(m, s.shape[-1], mode="linear", align_corners=False)
        log_p, logdet = self.forward(m)
        self.ℒ.log_p = -torch.mean(log_p)
        self.ℒ.logdet = -torch.mean(logdet)
        ℒ = self.ℒ.log_p + self.ℒ.logdet
        return ℒ
