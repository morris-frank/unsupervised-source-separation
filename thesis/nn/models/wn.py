from typing import Tuple

import torch
from torch.nn import functional as F

from . import BaseModel
from .umix import q_sǀm
from ...dist import AffineBeta
from ...utils import clean_init_args
from ..modules import MelSpectrogram


class WN(BaseModel):
    def __init__(self, mel_channels: int = 80, width: int = 64):
        super(WN, self).__init__()
        self.params = clean_init_args(locals().copy())
        self.name = ""

        self.n_classes = 4

        # The encoders
        self.q_sǀm = q_sǀm(mel_channels, width)

        # The placeholder for the prior networks
        self.p_s = None

        self.mel = MelSpectrogram()

    def forward(self, m: torch.Tensor, m_mel: torch.Tensor):
        m_mel = F.interpolate(m_mel, m.shape[-1], mode="linear", align_corners=False)

        α, β = self.q_sǀm(m, m_mel)
        q_s = AffineBeta(α, β)

        ŝ = q_s.rsample()
        log_q_ŝ = q_s.log_prob(ŝ).clamp(-1e5, 1e5)
        ŝ = F.normalize(ŝ, p=float("inf"), dim=-1)  # Max-Norm

        with torch.no_grad():
            ŝ_mel = self.mel(ŝ[:, 0, :])
            log_p_ŝ, _ = self.p_s(ŝ, ŝ_mel)
            log_p_ŝ = log_p_ŝ.detach()[:, None].clamp(-1e5, 1e5)

        self.ℒ.KL = -torch.mean(log_p_ŝ - log_q_ŝ)

        return ŝ

    def test(
        self, x: Tuple[torch.Tensor, torch.Tensor], s: torch.Tensor
    ) -> torch.Tensor:
        s, s_mel = x

        s_noised = (s + 0.1 * torch.randn_like(s)).clamp(-1, 1)
        s_mel_noised = self.mel(s_noised)
        ŝ = self.forward(s_noised, s_mel_noised)

        self.ℒ.supervision = F.mse_loss(ŝ, s)

        ℒ = self.ℒ.supervision
        return ℒ
