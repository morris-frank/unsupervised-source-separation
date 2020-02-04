from typing import Optional

import torch
from torch import nn
from torch.autograd import Variable

from .wavenet import Wavenet
from ..modules import ChannelConvInvert
from ...utils import clean_init_args


class WaveGlow(nn.Module):
    def __init__(self, channels: int, n_flows: int = 15, wn_layers: int = 12,
                 wn_width: int = 32):
        super(WaveGlow, self).__init__()
        self.params = clean_init_args(locals().copy())
        self.channels = channels
        self.n_flows = n_flows

        self.conv = nn.ModuleList()
        self.waves = nn.ModuleList()
        for _ in range(n_flows):
            self.conv.append(ChannelConvInvert(channels))
            self.waves.append(Wavenet(
                in_channels=channels // 2, out_channels=channels,
                c_channels=1, n_blocks=1, n_layers=wn_layers,
                residual_width=2 * wn_width, skip_width=wn_width)
            )

    def forward(self, m: torch.Tensor, S: torch.Tensor):
        assert S.shape[1] == self.channels

        f_s = S

        total_log_s, total_det_w = 0, 0
        for k in range(self.n_flows):
            # First mix the channels
            f_s, log_det_w = self.conv[k](f_s)

            # Separate them and get scale and translate
            s_a, s_b = f_s.chunk(2, dim=1)
            log_s_t = self.waves[k](s_a, m)
            log_s, t = log_s_t.chunk(2, dim=1)

            # Affine transform right part
            s_b = log_s.exp() * s_b + t

            # Merge them back
            f_s = torch.cat([s_a, s_b], 1)

            # Save for loss
            total_det_w = log_det_w + total_det_w
            total_log_s = log_s.sum() + total_log_s
        z = f_s
        return z, total_log_s, total_det_w

    def infer(self, m: torch.Tensor, σ: float = 1.,
              z: Optional[torch.Tensor] = None) -> torch.Tensor:
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

    @staticmethod
    def loss(σ: float = 1.):
        def func(model, x, y, progress):
            _ = progress
            z, total_log_s, total_det_w = model(x, y)
            loss = (z * z).sum() / (2 * σ ** 2) - total_log_s - total_det_w
            return loss / z.numel()

        return func
