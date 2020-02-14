from typing import Optional

import torch
from torch import nn
from torch.autograd import Variable

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
                    c_channels=1,
                    n_blocks=1,
                    n_layers=wn_layers,
                    residual_width=2 * wn_width,
                    skip_width=wn_width,
                )
            )

    def forward(self, m: torch.Tensor, S: torch.Tensor):
        assert S.shape[1] == self.channels

        f_s = S

        ℒ_log_s, ℒ_det_W = 0, 0
        for k in range(self.n_flows):
            # First mix the channels
            f_s, log_det_W = self.conv[k](f_s)

            # Separate them and get scale and translate
            s_a, s_b = f_s.chunk(2, dim=1)
            log_s_t = self.waves[k](s_a, m)
            log_s, t = log_s_t.chunk(2, dim=1)

            # Affine transform right part
            s_b = log_s.exp() * s_b + t

            # Merge them back
            f_s = torch.cat([s_a, s_b], 1)

            # Save for loss
            ℒ_det_W += log_det_W
            ℒ_log_s += log_s.sum()

        self.ℒ.det_W = ℒ_det_W
        self.ℒ.log_s = ℒ_log_s
        z = f_s
        return z

    def infer(
        self, m: torch.Tensor, σ: float = 1.0, z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        N, _, L = m.shape

        # Sample a z
        if z is not None:
            f_z = z.type(m.dtype).to(m.device)
        else:
            f_z = torch.empty(N, self.channels, L).type(m.dtype).to(m.device)
            f_z = Variable(σ * f_z.normal_())

        for k in reversed(range(self.n_flows)):
            z_a, z_b = f_z.chunk(2, dim=1)

            log_s_t = self.waves[k](z_a, m)
            log_s, t = log_s_t.chunk(2, dim=1)
            z_b = (z_b - t) / (log_s.exp())

            f_z = torch.cat([z_a, z_b], 1)

            f_z = self.conv[k](f_z, reverse=True)

        return f_z

    def test(self, m: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        σ = 1.0
        z = self(m, S)
        self.ℒ.z = (z * z).sum() / (2 * σ ** 2)
        ℒ = self.ℒ.z - self.ℒ.log_s - self.ℒ.det_W
        ℒ /= z.numel()
        return ℒ
