from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import MelSpectrogram

from . import BaseModel
from ..modules import MelSpectrogram
from ..wavenet import Wavenet
from ...dist import AffineBeta
from ...functional import normalize
from ...utils import clean_init_args


class q_sǀm(nn.Module):
    def __init__(self, width, mel_channels):
        super(q_sǀm, self).__init__()
        self.f = Wavenet(
            in_channels=1,
            out_channels=256,
            n_blocks=3,
            n_layers=11,
            residual_channels=width,
            gate_channels=width,
            skip_channels=width,
            cin_channels=mel_channels,
            bias=False,
            fc_kernel_size=3,
            fc_channels=2048,
        )

        self.f_α = nn.Sequential(nn.Conv1d(256, 1, 1, bias=False), nn.Softplus())
        self.f_β = nn.Sequential(nn.Conv1d(256, 1, 1, bias=False), nn.Softplus())

    def forward(self, m, m_mel=None):
        f = self.f(m, m_mel)
        α = self.f_α(f) + 1e-4
        β = self.f_β(f) + 1e-4
        return α, β


class Demixer(BaseModel):
    def __init__(
        self, n_classes: int = 4, width: int = 64, mel_channels: int = 80, **kwargs
    ):
        super(Demixer, self).__init__(**kwargs)
        self.params = clean_init_args(locals().copy())
        self.n_classes = n_classes

        # The encoders
        self.q_sǀm = nn.ModuleList()
        for k in range(self.n_classes):
            self.q_sǀm.append(q_sǀm(width, mel_channels))

        # The placeholder for the prior networks
        self.p_s = None

        # the decoder
        # self.p_mǀs = Wavenet(
        #     in_channels=self.n_classes,
        #     out_channels=1,
        #     n_blocks=1,
        #     n_layers=8,
        #     residual_channels=32,
        #     gate_channels=32,
        #     skip_channels=32,
        #     cin_channels=None,
        # )

        self.mel = MelSpectrogram()

    def q_s(self, m, m_mel):
        m_mel = F.interpolate(m_mel, m.shape[-1], mode="linear", align_corners=False)

        α, β = zip(*[q(m, m_mel) for q in self.q_sǀm])
        α, β = torch.cat(α, dim=1), torch.cat(β, dim=1)
        q_s = AffineBeta(α, β)
        return q_s

    def test_forward(self, m, m_mel):
        q_s = self.q_s(m, m_mel)
        ŝ = q_s.mean
        log_q_ŝ = q_s.log_prob(ŝ)

        ŝ = F.normalize(ŝ, p=float("inf"), dim=-1)

        p_ŝ = []
        for k in range(self.n_classes):
            ŝ_mel = self.mel(ŝ[:, k, :])
            log_p_ŝ, _ = self.p_s[k](ŝ[:, None, k, :], ŝ_mel)
            p_ŝ.append(log_p_ŝ)

        m_ = ŝ.mean(dim=1)
        return ŝ, m_, log_q_ŝ, torch.cat(p_ŝ, dim=1), q_s.α, q_s.β

    def forward(self, m, m_mel):
        q_s = self.q_s(m, m_mel)
        ŝ = q_s.rsample()
        log_q_ŝ = q_s.log_prob(ŝ).clamp(-1e5, 1e5)

        # Scale the posterior samples so they fill range [-1, 1].
        # This is necessary as we start around zero with the values and the
        # prior distributions assign too high likelihoods around zero!
        scaled_ŝ = normalize(ŝ)

        for k in range(self.n_classes):
            # Get Log likelihood under prior
            ŝ_mel = self.mel(scaled_ŝ[:, k, :])
            log_p_ŝ, _ = self.p_s[k](scaled_ŝ[:, None, k, :], ŝ_mel)
            log_p_ŝ = log_p_ŝ[:, None].clamp(-1e5, 1e5)

            # Kullback Leibler for this k'th source
            KL_k = -torch.mean(log_p_ŝ - log_q_ŝ[:, k, :])
            setattr(self.ℒ, f"KL_{k}", KL_k)

        m_ = scaled_ŝ.mean(dim=1, keepdim=True)
        self.ℒ.reconstruction = F.mse_loss(m_, m)

        return scaled_ŝ, m_

    def test(
        self, x: Tuple[torch.Tensor, torch.Tensor], s: torch.Tensor
    ) -> torch.Tensor:
        m, m_mel = x
        ŝ, _ = self.forward(m, m_mel)

        ℒ = self.ℒ.reconstruction
        for k in range(self.n_classes):
            ℒ += getattr(self.ℒ, f"KL_{k}")

        self.ℒ.l1_s = F.l1_loss(ŝ, s)
        # ℒ += self.ℒ.l1_s

        return ℒ
