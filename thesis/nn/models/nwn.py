import torch
from torch import nn
from torch.nn import functional as F

from . import BaseModel
from ..modules import MelSpectrogram
from ..wavenet import Wavenet
from ...utils import clean_init_args


class NWN(BaseModel):
    def __init__(self, width: int = 64):
        super(NWN, self).__init__()
        self.params = clean_init_args(locals().copy())
        self.name = ""

        self.n_classes = 4
        odim = 256

        # The encoders
        self.q_sǀm = self.f = nn.Sequential(
            Wavenet(
                in_channels=1,
                out_channels=odim,
                n_blocks=3,
                n_layers=11,
                residual_channels=width,
                gate_channels=width,
                skip_channels=width,
                bias=False,
                alternative=True,
            ),
            nn.Conv1d(odim, 1, 1, bias=False),
        )

        # The placeholder for the prior networks
        self.p_s = [None]

        self.mel = MelSpectrogram()

    def forward(self, m: torch.Tensor):
        ŝ = self.q_sǀm(m)

        with torch.no_grad():
            ŝ_ = F.normalize(ŝ, p=float("inf"), dim=-1)  # Max-Norm
            ŝ_mel = self.mel(ŝ_[:, 0, :])
            log_p_ŝ, _ = self.p_s[0](ŝ_, ŝ_mel)
            log_p_ŝ = log_p_ŝ.detach()[:, None].clamp(-1e5, 1e5)

        self.ℒ.likelihood = -torch.mean(log_p_ŝ)

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
