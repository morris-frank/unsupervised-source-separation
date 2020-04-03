import torch
from torch import nn
from torch.nn import functional as F

from . import BaseModel
from .umix import q_sǀm
from ..modules import MelSpectrogram
from ..wavenet import Wavenet
from ...dist import AffineBeta
from ...utils import clean_init_args


class q_sǀm(nn.Module):
    def __init__(self, dim, odim=256):
        super(q_sǀm, self).__init__()
        self.f = Wavenet(
            in_channels=1,
            out_channels=odim,
            n_blocks=3,
            n_layers=11,
            residual_channels=dim,
            gate_channels=dim,
            skip_channels=dim,
            bias=False,
            alternative=True,
        )

        self.f_α = nn.Sequential(nn.Conv1d(odim, 1, 1, bias=False), nn.Softplus())
        self.f_β = nn.Sequential(nn.Conv1d(odim, 1, 1, bias=False), nn.Softplus())

    def forward(self, m: torch.Tensor):
        f = self.f(m)
        α = self.f_α(f) + 1e-4
        β = self.f_β(f) + 1e-4
        return α, β


class WN(BaseModel):
    def __init__(self, width: int = 64):
        super(WN, self).__init__()
        self.params = clean_init_args(locals().copy())
        self.name = ""

        self.n_classes = 4

        # The encoders
        self.q_sǀm = q_sǀm(width)

        # The placeholder for the prior networks
        self.p_s = [None]

        self.mel = MelSpectrogram()

    def forward(self, m: torch.Tensor):
        α, β = self.q_sǀm(m)
        q_s = AffineBeta(α, β)

        ŝ = q_s.rsample()
        log_q_ŝ = q_s.log_prob(ŝ).clamp(-1e5, 1e5)

        with torch.no_grad():
            ŝ_ = F.normalize(ŝ, p=float("inf"), dim=-1)  # Max-Norm
            ŝ_mel = self.mel(ŝ_[:, 0, :])
            log_p_ŝ, _ = self.p_s[0](ŝ_, ŝ_mel)
            log_p_ŝ = log_p_ŝ.detach()[:, None].clamp(-1e5, 1e5)

        self.ℒ.KL = -torch.mean(log_p_ŝ - log_q_ŝ)

        return ŝ

    def test(self, s):
        s_noised = (s + 0.1 * torch.randn_like(s)).clamp(-1, 1)
        z = s_noised - s

        ŝ = self.forward(s_noised)
        ẑ = s_noised - ŝ

        self.ℒ.l1_s = F.l1_loss(ŝ, s)
        self.ℒ.l1_z = F.l1_loss(ẑ, z)

        ℒ = self.ℒ.l1_s + self.ℒ.l1_z
        return ℒ
