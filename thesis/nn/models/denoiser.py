import torch
from torch.nn import functional as F

from . import BaseModel
from .demixer import q_sǀm
from ..modules import MelSpectrogram
from ...dist import AffineBeta
from ...utils import clean_init_args


class Denoiser(BaseModel):
    def __init__(self, width: int = 64, **kwargs):
        super(Denoiser, self).__init__(**kwargs)
        self.params = clean_init_args(locals().copy())

        self.q_sǀm = q_sǀm(width, None)

        # The placeholder for the prior networks
        self.p_s = [None]

        self.mel = MelSpectrogram()

    def forward(self, m: torch.Tensor):
        α, β = self.q_sǀm(m)
        q_s = AffineBeta(α, β)
        ŝ = q_s.rsample()
        log_q_ŝ = q_s.log_prob(ŝ).clamp(-1e5, 1e5)

        with torch.no_grad():
            scaled_ŝ = F.normalize(ŝ, p=float("inf"), dim=-1)
            scaled_ŝ_mel = self.mel(scaled_ŝ[:, 0, :])
            log_p_ŝ, _ = self.p_s[0](scaled_ŝ, scaled_ŝ_mel)
            log_p_ŝ = log_p_ŝ.detach()[:, None].clamp(-1e5, 1e5)

        self.ℒ.KL = -torch.mean(log_p_ŝ - log_q_ŝ)

        return ŝ

    def test(self, s):
        s_noised = (s + 0.3 * torch.randn_like(s)).clamp(-1, 1)
        z = s_noised - s

        ŝ = self.forward(s_noised)
        ẑ = s_noised - ŝ

        self.ℒ.l1_s = F.l1_loss(ŝ, s)
        self.ℒ.l1_z = F.l1_loss(ẑ, z)

        ℒ = self.ℒ.l1_s + self.ℒ.l1_z
        return ℒ


class GAN(Denoiser):
    def test(self, s):
        z = torch.randn_like(s).clamp(-1, 1)
        _ = self.forward(z)
        ℒ = self.ℒ.KL
        return ℒ
