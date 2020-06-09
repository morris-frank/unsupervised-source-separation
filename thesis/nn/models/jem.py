from typing import Optional

import torch
from torch import Tensor as T
from torch import autograd
from torch import nn
from torch.nn import functional as F

from . import BaseModel
from ..wavenet import Wavenet
from ...utils import clean_init_args
from ...io import vprint


class JEM(BaseModel):
    def __init__(
        self, in_channels: int, out_channels: int, width: int, cin_channel: int = None
    ):
        super(JEM, self).__init__()
        self.params = clean_init_args(locals().copy())

        self.classes = out_channels

        self.ρ = 0.05  # Reinatialization probability for the Buffer
        self.η = 20  # Steps of internal SGLD
        self.α = 1  # SGLD step size / learning rate
        self.σ = 0.01  # SGLD added noise variance

        self.replay_buffer = self.init_random(1000)

        self.classify = nn.Sequential(
            Wavenet(
                in_channels=in_channels,
                out_channels=out_channels,
                n_blocks=1,
                n_layers=2,
                residual_channels=width,
                gate_channels=width,
                skip_channels=width,
                kernel_size=3,
                cin_channels=cin_channel,
                causal=False,
                zero_final=False,
                bias=False,
            ),
            nn.Linear(13, 1, bias=False),
        )

    def forward(self, s: T, i: Optional[T] = None) -> T:
        logits = self.classify(s)
        if i is None:
            return logits.logsumexp(1)
        else:
            return torch.gather(logits, 1, i[:, None, None])

    def test(self, s: T, i: T) -> T:
        # p(i|s)
        # Normal Cross-Entropy classification loss
        ī = self.classify(s)
        self.ℒ.p_iǀs = F.cross_entropy(ī.squeeze(), i.long())

        # p(s)
        # Sample from ŝ ~ p(s) with SGLD and E[s] = -LogSumExp_i f_θ(s)[i]
        self.eval()
        ŝ, î, b_i = self.sample_from_buffer(ī.shape[0], s.device)

        # SGLD
        ŝ = autograd.Variable(ŝ, requires_grad=True)
        for _ in range(self.η):
            δf_δŝ = autograd.grad(self(ŝ, i=î).sum(), [ŝ], retain_graph=True)[0]
            ŝ.data += self.α * δf_δŝ + self.σ * torch.randn_like(ŝ)
        self.train()
        ŝ = ŝ.detach()
        if len(self.replay_buffer) > 0:
            self.replay_buffer[b_i] = ŝ.cpu()

        self.ℒ.p_s = -(self(s).mean() - self(ŝ).mean())

        ℒ = self.ℒ.p_iǀs + self.ℒ.p_s
        return ℒ

    @staticmethod
    def init_random(bs: int):
        return torch.FloatTensor(bs, 80, 13).uniform_(-1, 1).cpu()

    def sample_from_buffer(self, bs, device):
        if len(self.replay_buffer) == 0:
            return self.init_random(bs), None
        î = torch.randint(0, self.classes, (bs,)).to(device)
        buffer_size = len(self.replay_buffer) // self.classes
        b_i = torch.randint(0, buffer_size, (bs,))
        b_i = î.cpu() * buffer_size + b_i
        ŝ_buffer = self.replay_buffer[b_i]
        ŝ_random = self.init_random(bs)
        where = (torch.rand(bs) < self.ρ).float()[:, None, None]
        ŝ = where * ŝ_random + (1 - where) * ŝ_buffer
        return ŝ.to(device), î, b_i
