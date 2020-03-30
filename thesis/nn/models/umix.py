from random import random
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import MelSpectrogram

from . import BaseModel
from ..wavenet import Wavenet
from ...dist import AffineBeta
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

        self.f_α = nn.Sequential(nn.Conv1d(dim, 1, 1), nn.Softplus())
        self.f_β = nn.Sequential(nn.Conv1d(dim, 1, 1), nn.Softplus())

    def forward(self, m: torch.Tensor, m_mel: torch.Tensor):
        f = self.f(m, m_mel)
        α = self.f_α(f) + 1e-4
        β = self.f_β(f) + 1e-4
        return α, β


class UMixer(BaseModel):
    def __init__(self, mel_channels: int = 80, width: int = 64):
        super(UMixer, self).__init__()
        self.params = clean_init_args(locals().copy())
        self.name = ""

        self.n_classes = 4

        # The encoders
        self.q_sǀm = nn.ModuleList()
        for k in range(self.n_classes - 1):
            self.q_sǀm.append(q_sǀm(mel_channels, width))

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

        n_fft = 1024
        hop_length = 256
        sr = 16000
        self.mel = MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=mel_channels,
            f_min=125,
            f_max=7600,
        )

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

        ŝ = F.normalize(ŝ, p=float('inf'), dim=-1)

        p_ŝ = []
        for k in range(self.n_classes):
            ŝ_mel = self.mel(ŝ[:, k, :])
            log_p_ŝ, _ = self.p_s[k](ŝ[:, None, k, :], ŝ_mel)
            p_ŝ.append(log_p_ŝ)

        m_ = ŝ.mean(dim=1)
        return ŝ, m_, log_q_ŝ, torch.cat(p_ŝ, dim=1), q_s.α, q_s.β

    def forward(
        self, m: torch.Tensor, m_mel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_s = self.q_s(m, m_mel)
        ŝ = q_s.rsample()
        log_q_ŝ = q_s.log_prob(ŝ).clamp(-1e5, 1e5)

        # Scale the posterior samples so they fill range [-1, 1].
        # This is necessary as we start around zero with the values and the
        # prior distributions assign too high likelihoods around zero!
        ŝ = F.normalize(ŝ, p=float('inf'), dim=-1)

        for k in range(self.n_classes):
            # Get Log likelihood under prior
            with torch.no_grad():
                ŝ_mel = self.mel(ŝ[:, k, :])
                log_p_ŝ, _ = self.p_s[k](ŝ[:, None, k, :], ŝ_mel)
                log_p_ŝ = log_p_ŝ.detach()[:, None].clamp(-1e5, 1e5)

            # Kullback Leibler for this k'th source
            KL_k = -torch.mean(log_p_ŝ - log_q_ŝ[:, k, :])
            setattr(self.ℒ, f"KL_{k}", KL_k)

        m_ = ŝ.mean(dim=1)
        self.ℒ.reconstruction = F.mse_loss(m_, m)

        return ŝ, m_

    def test(
        self, x: Tuple[torch.Tensor, torch.Tensor], s: torch.Tensor
    ) -> torch.Tensor:
        m, m_mel = x
        ŝ, _ = self.forward(m, m_mel)

        ℒ = self.ℒ.reconstruction
        for k in range(self.n_classes):
            ℒ += 1.0 * getattr(self.ℒ, f"KL_{k}")

        if random() < .3:
            self.ℒ.supervised_mse = F.mse_loss(ŝ, s)
            ℒ += self.ℒ.supervised_mse

        return ℒ
