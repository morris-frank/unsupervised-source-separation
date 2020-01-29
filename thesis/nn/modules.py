import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from .functions import VectorQuantizationStraightThrough, VectorQuantization


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super(VQEmbedding, self).__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1. / K, 1. / K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 1).contiguous()
        latents = VectorQuantization.apply(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 1).contiguous()
        z_q_x_, indices = VectorQuantizationStraightThrough.apply(
            z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 2, 1).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
                                               dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 2, 1).contiguous()

        return z_q_x, z_q_x_bar


class ChannelConvInvert(nn.Module):
    def __init__(self, channels: int):
        super(ChannelConvInvert, self).__init__()
        self.conv = nn.Conv1d(channels, channels, 1, bias=False)

        # Init weights shall be orthonormal:
        # this guarantees |det(weights)| == 1 but not the sign
        weights = torch.qr(torch.empty(channels, channels).normal_())[0]

        if torch.det(weights) < 0:
            weights[:, 0] = -weights[:, 0]

        self.conv.weight.data = weights.unsqueeze(-1)
        self.inv_w = None

    def forward(self, z: torch.Tensor, reverse: bool = False):
        [n, _, l] = z.size()

        w = self.conv.weight.squeeze()

        if reverse:
            if self.inv_w is None:
                self.inv_w = Variable(w.type(z.dtype).inverse().unsqueeze(-1))
            z = F.conv1d(z, self.inv_w, bias=None)
            return z
        else:
            logdet_w = n * l * torch.logdet(w)
            z = self.conv(z)

            return z, logdet_w
