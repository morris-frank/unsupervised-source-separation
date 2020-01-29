import torch
from torch import nn
from torch.autograd import Variable

from .wavenet import Wavenet
from ..modules import ChannelConvInvert
from ...utils import clean_init_args


# TODO: add skip connections / early outputs
class WaveGlow(nn.Module):
    def __init__(self, channels: int, n_flows: int = 10):
        super(WaveGlow, self).__init__()
        self.params = clean_init_args(locals().copy())
        self.channels = channels
        self.n_flows = n_flows

        wn_layers, wn_width = 12, 32

        self.conv = nn.ModuleList()
        self.waves = nn.ModuleList()
        for _ in range(n_flows):
            self.conv.append(ChannelConvInvert(channels))
            self.waves.append(Wavenet(
                in_channels=channels, out_channels=channels, n_blocks=1,
                n_layers=wn_layers, residual_width=2 * wn_width,
                skip_width=wn_width, conditional_dims=[(1, 1)])
            )

    def forward(self, mix: torch.Tensor, sources: torch.Tensor):
        assert sources.shape[1] == self.channels

        m = mix
        f_s = sources

        total_log_s, total_det_w = 0, 0
        for k in range(self.n_flows):
            # First mix the channels
            f_s, log_det_w = self.conv[k](f_s)

            # Separate them and get scale and translate
            s_a, s_b = f_s.chunk(2, dim=1)
            log_s_t = self.waves[k](s_a, m)
            log_s, t = log_s_t.chunck(2, dim=1)

            # Affine transform right part
            s_b = log_s.exp() * s_a + t

            # Merge them back
            f_s = torch.cat([s_a, s_b], 1)

            # Save for loss
            total_det_w = log_det_w + total_det_w
            total_log_s = log_s.sum() + total_log_s
        z = f_s
        return z, total_log_s, total_det_w

    def infer(self, mix: torch.Tensor, σ: float = 1.):
        N, _, L = mix.shape

        # Sample a z
        f_z = torch.empty(N, self.channels, L).type(mix.dtype).to(mix.device)
        f_z = Variable(σ * f_z.normal_())

        for k in reversed(range(self.n_flows)):
            z_a, z_b = f_z.chunk(2, dim=1)

            log_s_t = self.waves[k](z_a, mix)
            log_s, t = log_s_t.chunk(2, dim=1)
            z_b = (z_a - t) / log_s.exp()

            f_z = torch.cat([z_a, z_b], 1)

            f_z = self.conv[k](f_z, reverse=True)

        return f_z

    @staticmethod
    def loss(σ: float = 1.):
        def func(model, x, y, progress):
            _ = progress
            z, total_log_s, total_det_w = model(x, y)
            loss = (z * z).sum() / (2 * σ ** 2) - total_log_s - total_det_w
            return loss

        return func