from math import log, pi

import torch
from torch import nn
from torch.nn import functional as F

from . import BaseModel
from ..wavenet import Wavenet
from ...functional import likelihood_normal, split_LtoC, flip_channels
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


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, cin_channel, width=256, num_layer=6):
        super().__init__()

        self.net = Wavenet(
            in_channels=in_channel // 2,
            out_channels=in_channel,
            n_blocks=1,
            n_layers=num_layer,
            residual_channels=width,
            gate_channels=width,
            skip_channels=width,
            kernel_size=3,
            cin_channels=cin_channel // 2,
            causal=False,
            zero_final=True,
        )

    def forward(self, x, c=None):
        in_a, in_b = x.chunk(2, 1)
        c_a, c_b = c.chunk(2, 1)

        log_s, t = self.net(in_a, c_a).chunk(2, 1)

        out_b = (in_b - t) * torch.exp(-log_s)
        logdet = torch.sum(-log_s)

        return torch.cat([in_a, out_b], 1), logdet


class Flow(nn.Module):
    def __init__(
        self, in_channel, cin_channel, width, num_layer,
    ):
        super().__init__()

        self.actnorm = ActNorm(in_channel)
        self.coupling = AffineCoupling(
            in_channel, cin_channel, width=width, num_layer=num_layer
        )

    def forward(self, x, c=None):
        out, logdet = self.actnorm(x)
        out, det = self.coupling(out, c)
        out = flip_channels(out)
        c = flip_channels(c)

        if det is not None:
            logdet = logdet + det

        return out, c, logdet


class Block(nn.Module):
    def __init__(
        self, in_channel, cin_channel, n_flow, n_layer, width, split=False,
    ):
        super().__init__()

        self.split = split
        squeeze_dim = in_channel * 2
        squeeze_dim_c = cin_channel * 2

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(
                Flow(squeeze_dim, squeeze_dim_c, width=width, num_layer=n_layer,)
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
                cin_channels=squeeze_dim_c,
                causal=False,
                zero_final=True,
            )

    def forward(self, x, c):
        x = split_LtoC(x)
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


class Flowavenet(BaseModel):
    def __init__(
        self,
        in_channel,
        cin_channel,
        n_block,
        n_flow,
        n_layer,
        width,
        block_per_split,
        **kwargs
    ):
        super(Flowavenet, self).__init__(**kwargs)
        self.params = clean_init_args(locals().copy())
        self.block_per_split, self.n_block = block_per_split, n_block

        self.blocks = nn.ModuleList()
        for i in range(self.n_block):
            split = (i < self.n_block - 1) and (i + 1) % self.block_per_split == 0

            self.blocks.append(
                Block(
                    in_channel, cin_channel, n_flow, n_layer, split=split, width=width
                )
            )
            cin_channel *= 2
            if not split:
                in_channel *= 2

        self.upsample_conv = nn.ModuleList()
        for s in [16, 16]:
            convt = nn.ConvTranspose2d(
                1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s)
            )
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))

    def forward(self, x, c):
        B, _, T = x.size()
        logdet, log_p_sum = 0, 0
        out = x
        c = self.upsample(c)[:, :, :T]

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

    def upsample(self, c):
        c = c.unsqueeze(1)
        for f in self.upsample_conv:
            c = f(c)
        c = c.squeeze(1)
        return c

    def test(self, source, mel):
        B, _, T = source.size()
        log_p, logdet = self.forward(source, mel)
        self.ℒ.log_p, self.ℒ.logdet = -torch.mean(log_p), -torch.mean(logdet)
        ℒ = self.ℒ.log_p + self.ℒ.logdet
        return ℒ
