import torch
from torch import nn
from torch import Tensor as T
from torch.nn import functional as F
from ..modules import ZeroConv2d, InvConv2d
from ...functional import chunk, interleave
from ...dist import norm_log_prob


class ActNorm(nn.Module):
    def __init__(self, in_channel, pretrained=False):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.initialized = pretrained

    def initialize(self, x: T):
        with torch.no_grad():
            flatten = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, x: T):
        if not self.initialized:
            if self.training:
                self.initialize(x)

        _, _, H, W = x.shape
        log_abs = self.scale.abs().log()
        log_det = torch.sum(log_abs) * H * W

        return self.scale * (x + self.loc), log_det

    def reverse(self, y):
        return y / self.scale - self.loc


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, groups=1):
        super().__init__()

        self.groups = groups

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, x: T):
        in_a, in_b = chunk(x, groups=self.groups)

        log_s, t = chunk(self.net(in_a), groups=self.groups)
        s = F.sigmoid(log_s + 2)
        out_b = (in_b + t) * s

        log_det = torch.sum(torch.log(s).view(x.shape[0], -1), 1)

        return interleave((in_a, out_b), groups=self.groups), log_det

    def reverse(self, y):
        out_a, out_b = chunk(y, groups=self.groups)

        log_s, t = chunk(self.net(out_a), groups=self.groups)
        s = F.sigmoid(log_s + 2)
        in_b = out_b / s - t

        return interleave((out_a, in_b), groups=self.groups)


class Flow(nn.Module):
    def __init__(self, in_channel, groups=1):
        super().__init__()

        self.groups = groups

        self.actnorm = ActNorm(in_channel)
        self.invconv = InvConv2d(in_channel)
        self.coupling = AffineCoupling(in_channel, groups=groups)

    def forward(self, x):
        out, log_det = self.actnorm(x)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)

        log_det = log_det + det1
        if det2 is not None:
            log_det = log_det + det2

        return out, log_det

    def reverse(self, y):
        x = self.coupling.reverse(y)
        x = self.invconv.reverse(x)
        x = self.actnorm.reverse(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, groups=1):
        super().__init__()

        squeeze_dim = in_channel * 4
        self.groups = groups

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, groups=groups))

        self.split = split

        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)
        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, x):
        N, C, H, W = x.shape
        squeezed = x.view(N, C, H // 2, 2, W // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(N, C * 4, H // 2, W // 2)

        log_det = 0

        for flow in self.flows:
            out, det = flow(out)
            log_det = log_det + det

        if self.split:
            out, z_new = chunk(out, groups=self.groups)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = norm_log_prob(z_new, mean, log_sd)
            log_p = log_p.view(N, -1).sum(1)
        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = norm_log_prob(out, mean, log_sd)
            log_p = log_p.view(N, -1).sum(1)
            z_new = out

        return out, log_det, log_p, z_new

    def reverse(self, y, eps=None, reconstruct=False):
        x = y

        if reconstruct:
            if self.split:
                x = torch.cat([y, eps], 1)

            else:
                x = eps

        else:
            if self.split:
                mean, log_sd = self.prior(x).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                x = torch.cat([y, z], 1)

            else:
                zero = torch.zeros_like(x)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                x = z

        for flow in self.flows[::-1]:
            x = flow.reverse(x)

        b_size, n_channel, height, width = x.shape

        unsqueezed = x.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed


class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, groups=1):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, groups=groups))
            n_channel *= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, groups=groups))

    def forward(self, x):
        log_p_sum = 0
        log_det = 0
        out = x
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            log_det = log_det + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, log_det, z_outs

    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                x = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                x = block.reverse(x, z_list[-(i + 1)], reconstruct=reconstruct)

        return x