from typing import Tuple

import torch
from torch.nn import functional as F
from torchaudio.transforms import MelSpectrogram

from . import BaseModel
from .umix import q_sǀm
from ...dist import AffineBeta
from ...utils import clean_init_args


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

    def forward(self, m: torch.Tensor, m_mel: torch.Tensor):
        m_mel = F.interpolate(m_mel, m.shape[-1], mode="linear", align_corners=False)
        α, β = self.q_sǀm(m, m_mel)
        q_s = AffineBeta(α, β)
        ŝ = q_s.rsample()
        log_q_ŝ = q_s.log_prob(ŝ).clamp(-1e5, 1e5)
        ŝ = F.normalize(ŝ, p=float("inf"), dim=-1)

        with torch.no_grad():
            ŝ_mel = self.mel(ŝ[:, 0, :])
            log_p_ŝ, _ = self.p_s(ŝ, ŝ_mel)
            log_p_ŝ = log_p_ŝ.detach()[:, None].clamp(-1e5, 1e5)

        KL_k = -torch.mean(log_p_ŝ - log_q_ŝ)
        setattr(self.ℒ, f"KL", KL_k)
        return ŝ

    def test(
        self, x: Tuple[torch.Tensor, torch.Tensor], s: torch.Tensor
    ) -> torch.Tensor:
        m, m_mel = x
        _ = self.forward(torch.rand_like(m), torch.rand_like(m_mel))

        ℒ = self.ℒ.KL
        return ℒ
