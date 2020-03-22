from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from . import BaseModel
from ..wavenet import Wavenet
from ...functional import rsample_truncated_normal
from ...utils import clean_init_args


class q_sǀm(nn.Module):
    def __init__(self, mel_channels, dim):
        super(q_sǀm, self).__init__()
        self.f = Wavenet(
            in_channels=1,
            out_channels=dim,
            n_blocks=3,
            n_layers=11,
            residual_channels=dim,
            gate_channels=dim,
            skip_channels=dim,
            cin_channels=mel_channels,
        )

        self.f_α = nn.Sequential(nn.Conv1d(dim, 1, 1), nn.Tanh())
        self.f_β = nn.Sequential(nn.Conv1d(dim, 1, 1), nn.Softplus())

    def forward(self, m: torch.Tensor, m_mel: torch.Tensor):
        f = self.f(m, m_mel)
        α = self.f_α(f)
        β = self.f_β(f) + 1e-10
        return α, β


class NUMixer(BaseModel):
    def __init__(self, mel_channels: int = 80, width: int = 64):
        super(NUMixer, self).__init__()
        self.params = clean_init_args(locals().copy())
        self.name = "only_supervised_normal"

        self.n_classes = 4

        # The encoders
        self.q_sǀm = nn.ModuleList()
        for k in range(self.n_classes):
            self.q_sǀm.append(q_sǀm(mel_channels, width))

        self.upsample_conv = nn.ModuleList()
        for s in [16, 16]:
            convt = nn.ConvTranspose2d(
                1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s)
            )
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))

    def q_s(self, m, m_mel):
        m_mel = self.upsample(m_mel)
        m_mel = m_mel[:, :, : m.shape[-1]]

        α, β = zip(*[q(m, m_mel) for q in self.q_sǀm])
        α, β = torch.cat(α, dim=1), torch.cat(β, dim=1)
        return α, β

    def forward(
        self, m: torch.Tensor, m_mel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        μ, σ = self.q_s(m, m_mel)
        ŝ, log_q_ŝ = rsample_truncated_normal(μ, σ, ll=True)

        m_ = None
        return ŝ, m_

    def test(
        self, x: Tuple[torch.Tensor, torch.Tensor], s: torch.Tensor
    ) -> torch.Tensor:
        m, m_mel = x
        ŝ, _ = self.forward(m, m_mel)

        self.ℒ.supervised_l1_recon = F.l1_loss(ŝ, s)
        ℒ = self.ℒ.supervised_l1_recon
        return ℒ

    def upsample(self, c):
        c = c.unsqueeze(1)
        for f in self.upsample_conv:
            c = f(c)
        c = c.squeeze(1)
        return c
