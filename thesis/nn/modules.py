from typing import Optional

import torch
from torch import dtype as torch_dtype
from torch import nn

from .functions import VectorQuantizationStraightThrough, VectorQuantization


class DilatedQueue:
    def __init__(self, size: int, data: Optional[torch.Tensor] = None,
                 channels: int = 1, dilation: int = 1,
                 dtype: torch_dtype = torch.float32):
        self.idx_en, self.idx_de = 0, 0
        self.channels, self.size = channels, size
        self.dtype = dtype
        self.dilation = dilation

        self.data = data
        if data is None:
            self.reset()

    def enqueue(self, x):
        self.data[:, self.idx_en] = x
        self.idx_en = (self.idx_en + 1) % self.size

    def dequeue(self, num_deq=1):
        start = self.idx_de - ((num_deq - 1) * self.dilation)
        end = self.idx_de
        if start < 0:
            t1 = self.data[:, start::self.dilation]
            t2 = self.data[:, self.idx_de % self.dilation:end:self.dilation]
            t = torch.cat((t1, t2), 1)
        else:
            t = self.data[:, start:end:self.dilation]

        self.idx_de = (self.idx_de + 1) % self.size
        return t

    def reset(self, device: str = 'cpu'):
        self.idx_en, self.idx_de = 0, 0
        self.data = torch.zeros((self.channels, self.size),
                                dtype=self.dtype).to(device)


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
