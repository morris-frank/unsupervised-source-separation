from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from . import BaseModel
from ..modules import STFTUpsample
from ..wavenet import Wavenet
from ...dist import rsample_truncated_normal
from ...utils import clean_init_args


class q_sǀm(nn.Module):
    def __init__(self, mel_channels, dim, n_blocks=3):
        super(q_sǀm, self).__init__()
        self.f = Wavenet(
            in_channels=1,
            out_channels=dim,
            n_blocks=n_blocks,
            n_layers=11,
            residual_channels=dim,
            gate_channels=dim,
            skip_channels=dim,
            cin_channels=mel_channels,
        )

        self.f_μ = nn.Sequential(nn.Conv1d(dim, 1, 1), nn.Tanh())
        self.f_σ = nn.Sequential(nn.Conv1d(dim, 1, 1), nn.Softplus())

    def forward(self, m: torch.Tensor, m_mel: torch.Tensor):
        f = self.f(m, m_mel)
        μ = self.f_μ(f) * (1 - 1e-6)
        σ = self.f_σ(f) + 1e-10
        return μ, σ


class NUMixer(BaseModel):
    def __init__(self, mel_channels: int = 80, width: int = 64):
        super(NUMixer, self).__init__()
        self.params = clean_init_args(locals().copy())
        self.name = "only_supervised_normal"

        self.n_classes = 4

        # A learned upsampler for the conditional
        self.c_up = STFTUpsample([16, 16])

        # The encoders
        self.q_sǀm = nn.ModuleList()
        for k in range(self.n_classes):
            self.q_sǀm.append(q_sǀm(mel_channels, width))

    def q_s(self, m, m_mel):
        m_mel = self.c_up(m_mel, m.shape[-1])

        μ, σ = zip(*[q(m, m_mel) for q in self.q_sǀm])
        return torch.cat(μ, dim=1), torch.cat(σ, dim=1)

    def forward(self, m: torch.Tensor, m_mel: torch.Tensor):
        μ, σ = self.q_s(m, m_mel)
        ŝ, log_q_ŝ = rsample_truncated_normal(μ, σ, ll=True)
        return ŝ

    def test(
        self, x: Tuple[torch.Tensor, torch.Tensor], s: torch.Tensor
    ) -> torch.Tensor:
        m, m_mel = x
        ŝ = self.forward(m, m_mel)

        self.ℒ.supervised_mse = F.mse_loss(ŝ, s)
        ℒ = self.ℒ.supervised_mse
        return ℒ
