import torch
from torch import nn
from torch.nn import functional as F

from . import BaseModel
from ..modules import ChannelConvInvert
from ..wavenet import Wavenet
from ...utils import clean_init_args


class WaveGlow(BaseModel):
    def __init__(
        self, channels: int, n_flows: int = 15, wn_layers: int = 12, wn_width: int = 32
    ):
        super(WaveGlow, self).__init__()
        self.params = clean_init_args(locals().copy())
        self.channels = channels
        self.n_flows = n_flows

        self.conv = nn.ModuleList()
        self.waves = nn.ModuleList()
        for _ in range(n_flows):
            self.conv.append(ChannelConvInvert(channels))
            self.waves.append(
                Wavenet(
                    in_channels=channels // 2,
                    out_channels=channels,
                    n_blocks=1,
                    n_layers=wn_layers,
                    residual_width=2 * wn_width,
                    skip_width=wn_width,
                )
            )

    def forward(self, m: torch.Tensor):
        f_m = m.repeat(1, self.channels, 1)

        ℒ_log_s, ℒ_det_W = 0, 0
        for k in range(self.n_flows):
            # First mix the channels
            f_m, log_det_W = self.conv[k](f_m)

            # Separate them and get scale and translate
            m_a, m_b = f_m.chunk(2, dim=1)
            log_s_t = self.waves[k](m_a)
            log_s, t = log_s_t.chunk(2, dim=1)

            # Affine transform right part
            m_b = log_s.exp() * m_b + t

            # Merge them back
            f_m = torch.cat([m_a, m_b], 1)

            # Save for loss
            ℒ_det_W += log_det_W
            ℒ_log_s += log_s.sum()

        self.ℒ.det_W = ℒ_det_W
        self.ℒ.log_s = ℒ_log_s
        S_tilde = f_m
        return S_tilde

    def infer(self, m: torch.Tensor) -> torch.Tensor:
        return self.forward(m)

    def test(self, m: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        σ = 1.0
        α, β = 1., 1.
        S_tilde = self(m, S)
        self.ℒ.p_z_likelihood = (S_tilde * S_tilde).sum() / (2 * σ ** 2)
        self.ℒ.reconstruction = F.mse_loss(S_tilde, S)
        print()
        print()
        print()
        import ipdb; ipdb.set_trace()
        print()
        print()
        print()
        ℒ = α * self.ℒ.p_z_likelihood - self.ℒ.det_W - self.ℒ.log_s + β * self.ℒ.reconstruction
        return ℒ
