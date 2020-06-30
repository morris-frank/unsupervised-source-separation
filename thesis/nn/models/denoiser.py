import torch
from torch import Tensor as T
from torch.nn import functional as F

from . import BaseModel
from .demixer import q_sǀm
from ..modules import MelSpectrogram
from ...functional import normalize
from ...utils import clean_init_args


class Denoiser(BaseModel):
    def __init__(self, width: int = 64, **kwargs):
        super(Denoiser, self).__init__(**kwargs)
        self.params = clean_init_args(locals().copy())

        self.iteration = 0

        self.q_sǀm = q_sǀm(4, width, None)

        # The placeholder for the prior networks
        self.p_s = [None]

        self.spectrograph = MelSpectrogram(n_mels=80, sr=16000)

    def forward(self, m: T) -> T:
        q_s = self.q_sǀm(m)
        ŝ = q_s.rsample()
        log_q_ŝ = q_s.log_prob(ŝ).clamp(-1e4, 1e3)

        scaled_ŝ = normalize(ŝ)
        scaled_ŝ_mel = torch.cat([F.interpolate(self.spectrograph(scaled_ŝ[:, i, :]), m.shape[-1], mode="linear", align_corners=False) for i in range(4)], dim=1)
        _, log_p_ŝ, _ = self.p_s[0](scaled_ŝ_mel)
        log_p_ŝ = log_p_ŝ[:, None].clamp(-1e5, 1e5)

        self.ℒ.log_p = log_p_ŝ.mean()
        self.ℒ.log_q = log_q_ŝ.mean()
        self.ℒ.KL = -torch.mean(log_p_ŝ - log_q_ŝ)

        return scaled_ŝ

    def test(self, s: T) -> T:
        s_noised = (s + 0.3 * torch.randn_like(s)).clamp(-1, 1)
        z = s_noised - s

        ŝ = self.forward(s_noised)
        ẑ = s_noised - ŝ

        self.ℒ.l1_s = F.l2_loss(ŝ, s)
        self.ℒ.l1_z = F.l2_loss(ẑ, z)

        ℒ = self.ℒ.l1_s
        return ℒ


class Denoiser_Semi(Denoiser):
    def test(self, s):
        s_noised = (s + 0.3 * torch.randn_like(s)).clamp(-1, 1)
        z = s_noised - s

        ŝ = self.forward(s_noised)
        ẑ = s_noised - ŝ
        ℒ = self.ℒ.KL

        self.ℒ.l1_s = F.l1_loss(ŝ, s)
        self.ℒ.l1_z = F.l1_loss(ẑ, z)

        ℒ = -self.ℒ.log_p
        # if self.iteration < 20:
        #     ℒ += self.ℒ.l1_s + self.ℒ.l1_z
        self.iteration += 1
        return ℒ
