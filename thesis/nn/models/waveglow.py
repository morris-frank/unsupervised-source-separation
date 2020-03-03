import torch
from torch import nn
from torch.nn import functional as F

from . import BaseModel
from ..modules import ChannelConvInvert
from ..old_wavenet import Wavenet
from ...utils import clean_init_args


class WaveGlow(BaseModel):
    def __init__(
        self, channels: int, n_flows: int = 15, wn_layers: int = 12, wn_width: int = 32,
        stride_skip: int = 2
    ):
        super(WaveGlow, self).__init__()
        self.params = clean_init_args(locals().copy())
        self.channels = channels
        self.n_flows = n_flows
        self.stride_skip = stride_skip
        self.size_skip = channels // (n_flows // stride_skip)

        assert n_flows % stride_skip == 0

        self.conv = nn.ModuleList()
        self.waves = nn.ModuleList()

        rem_channels = channels
        for k in range(n_flows):
            self.conv.append(ChannelConvInvert(channels))
            self.waves.append(
                Wavenet(
                    in_channels=rem_channels // 2,
                    out_channels=rem_channels,
                    n_blocks=1,
                    n_layers=wn_layers,
                    residual_width=2 * wn_width,
                    skip_width=wn_width,
                    zero_final=True,
                )
            )

            if (k+1) % stride_skip == 0:
                rem_channels -= self.size_skip

    def forward(self, m: torch.Tensor):
        bs, _, l = m.shape

        f_m = m
        S_tilde = []

        ℒ_log_s, ℒ_det_W = 0, 0
        for k in range(self.n_flows):
            # Early exit of the highway of loooove
            if k % self.stride_skip == 0 and k > 0:
                S_tilde.append(f_m[: :self.size_skip, :])
                f_m = f_m[:, self.size_skip:, :]

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
            ℒ_log_s += log_s.mean()

        self.ℒ.det_W = -ℒ_det_W
        self.ℒ.log_s = -ℒ_log_s

        return torch.cat(S_tilde, 1)

    def infer(self, m: torch.Tensor) -> torch.Tensor:
        return self.forward(m)

    def test(self, m: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        σ = 1.0
        α, β = 1.0, 1.0
        self.S = S
        S_tilde = self.forward(m)
        self.ℒ.p_z_likelihood = α * (S_tilde ** 2).mean() / (2 * σ ** 2)
        self.ℒ.reconstruction = β * F.mse_loss(S_tilde, S)
        ℒ = self.ℒ.p_z_likelihood + self.ℒ.det_W + self.ℒ.log_s + self.ℒ.reconstruction
        return ℒ
