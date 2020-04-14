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
    def __init__(self, n_classes, width, mel_channels):
        super(q_sǀm, self).__init__()
        out_channels = 256

        self.f = Wavenet(
            in_channels=n_classes,
            out_channels=out_channels * n_classes,
            n_blocks=3,
            n_layers=11,
            residual_channels=width * n_classes,
            gate_channels=width * n_classes,
            skip_channels=width * n_classes,
            cin_channels=mel_channels,
            bias=False,
            fc_kernel_size=3,
            fc_channels=2048,
            groups=n_classes
        )

        self.f_α = nn.Sequential(nn.Conv1d(out_channels * n_classes, n_classes, 1, bias=False, groups=n_classes), nn.Softplus())
        self.f_β = nn.Sequential(nn.Conv1d(out_channels * n_classes, n_classes, 1, bias=False, groups=n_classes), nn.Softplus())

    def forward(self, m, m_mel=None):
        m_mel = F.interpolate(m_mel, m.shape[-1], mode="linear",
                              align_corners=False)
        f = self.f(m, m_mel)
        α = self.f_α(f) + 1e-4
        β = self.f_β(f) + 1e-4
        q_s = AffineBeta(α, β)
        return q_s


class Demixer(BaseModel):
    def __init__(
        self, n_classes: int = 4, width: int = 64, mel_channels: int = 80, **kwargs
    ):
        super(Demixer, self).__init__(**kwargs)
        self.params = clean_init_args(locals().copy())
        self.n_classes = n_classes

        # The encoders
        self.q_sǀm = q_sǀm(n_classes, width, mel_channels)

        # The placeholder for the prior networks
        self.p_s = None

        self.mel = MelSpectrogram()
        self.iteration = 0

    def forward(self, m, m_mel):
        q_s = self.q_sǀm(m, m_mel)
        ŝ = q_s.rsample()
        log_q_ŝ = q_s.log_prob(ŝ).clamp(-1e5, 1e5)

        # Scale the posterior samples so they fill range [-1, 1].
        # This is necessary as we start around zero with the values and the
        # prior distributions assign too high likelihoods around zero!
        scaled_ŝ = normalize(ŝ)

        ŝ_mel = self.mel(scaled_ŝ)
        N, C, MC, L = ŝ_mel.shape
        ŝ_mel = ŝ_mel.view(N, C*MC, L)

        p_ŝ = self.p_s(scaled_ŝ, ŝ_mel)
        log_p_ŝ = p_ŝ.clamp(min=1e-12).log()

        self.ℒ.KL = -torch.mean(log_p_ŝ - log_q_ŝ)

        # for k in range(self.n_classes):
        #     # Get Log likelihood under prior
        #     ŝ_mel = self.mel(scaled_ŝ[:, k, :])
        #     log_p_ŝ, _ = self.p_s[k](scaled_ŝ[:, None, k, :], ŝ_mel)
        #     log_p_ŝ = log_p_ŝ[:, None].clamp(-1e5, 1e5)
        #
        #     # Kullback Leibler for this k'th source
        #     KL_k = -torch.mean(log_p_ŝ - log_q_ŝ[:, k, :])
        #     setattr(self.ℒ, f"KL/{k}", KL_k)

        m_ = ŝ.mean(dim=1, keepdim=True)

        return ŝ, m_

    def test(
        self, x: Tuple[torch.Tensor, torch.Tensor], s: torch.Tensor
    ) -> torch.Tensor:
        # sigmas = [0.5000822, 1.00011822, 0.33878675, 0.3480723]
        m, m_mel = x
        # β = min(self.iteration/500, 1)

        ŝ, m_ = self.forward(m.repeat(1, 4, 1), m_mel)
        self.ℒ.reconstruction = F.mse_loss(m_, m)

        # for k in range(self.n_classes):
        #     setattr(self.ℒ, f"variance/{k}", ((ŝ[:, k, :].var(-1) - sigmas[k])**2).mean())

        ℒ = self.ℒ.reconstruction + self.ℒ.KL
        # for k in range(self.n_classes):
        #     ℒ += β * getattr(self.ℒ, f"KL/{k}")
            # ℒ += getattr(self.ℒ, f"variance/{k}")

        # self.ℒ.l1_s = F.l1_loss(ŝ, s)
        # ℒ += self.ℒ.l1_s

        self.iteration += 1

        return ℒ
