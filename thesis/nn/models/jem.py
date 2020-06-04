from ..wavenet import Wavenet
from ...utils import clean_init_args
from . import BaseModel
from torch.nn import functional as F
import torch
from torch import autograd
from torch import Tensor as T


class JEM(BaseModel):
    def __init__(self, in_channels: int, out_channels: int, width: int, cin_channel: int = None):
        super(JEM, self).__init__()
        self.params = clean_init_args(locals().copy())

        self.ρ = 0.05       # Reinatialization probability for the Buffer
        self.η = 20         # Steps of internal SGLD
        self.α = 1          # SGLD step size / learning rate
        self.σ = 0.01       # SGLD added noise variance

        self.replay_buffer = self.init_random(1000)

        self.wave = Wavenet(
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
        )

    def forward(self, s: T) -> T:
        return self.wave(s)

    def test(self, s: T, i: T) -> T:
        bs = s.shape[0]

        # p(i|s)
        # Normal Cross-Entropy classification loss
        print(f"s {s.shape},\ti {i.shape}")
        ī = self.forward(s)
        self.ℒ.p_iǀs = F.cross_entropy(ī, i)

        # p(s)
        # Sample from ŝ ~ p(s) with SGLD and E[s] = -LogSumExp_i f_θ(s)[i]
        self.eval()
        ŝ, b_i = self.sample_from_buffer(bs, s.device)

        # SGLD
        ŝ = autograd.Variable(ŝ, requires_grad=True)
        for _ in range(self.η):
            δf_δŝ = autograd.grad(self(ŝ, i=i).sum(), [ŝ], retain_graph=True)[0]
            ŝ += self.α * δf_δŝ + self.σ * torch.randn_like(ŝ)
        self.replay_buffer.append(ŝ)
        self.train()
        ŝ = ŝ.detach()
        if len(self.replay_buffer) > 0:
            self.replay_buffer[b_i] = ŝ.cpu()

        self.ℒ.p_s = -(self(s).mean() - self(ŝ).mean())

        ℒ = self.ℒ.p_iǀs + self.ℒ.p_s
        return ℒ

    @staticmethod
    def init_random(bs: int):
        return torch.FloatTensor(bs, 1, 3072).uniform_(-1, 1).cpu()

    def sample_from_buffer(self, bs, device):
        if len(self.replay_buffer) == 0:
            return self.init_random(bs), None
        î = torch.randint(0, self.N, (bs,)).to(device)
        buffer_size = len(self.replay_buffer) // self.N
        b_i = torch.randint(0, buffer_size, (bs,))
        b_i = î.cpu() * buffer_size + b_i
        ŝ_buffer = self.replay_buffer[b_i]
        ŝ_random = self.init_random(bs)
        where = (torch.rand(bs) < self.ρ).float()[:, None, None, None]
        ŝ = where * ŝ_random + (1 - where) * ŝ_buffer
        return ŝ.to(device), b_i

