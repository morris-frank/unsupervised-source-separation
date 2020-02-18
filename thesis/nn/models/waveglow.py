import torch
from torch import nn
from torch.nn import functional as F

from . import BaseModel
from ..modules import ChannelConvInvert
from ..wavenet import Wavenet
from ...utils import clean_init_args


def plot(list_of_waves):
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    n = len(list_of_waves)
    l = 500

    fig, axs = plt.subplots(n, 1)
    for (name, wave), ax in zip(list_of_waves, axs):
        ax.set_title(name)
        for i in range(wave.shape[1]):
            ax.plot(range(l), wave[:, i, :l].squeeze())
    return fig


class WaveGlow(BaseModel):
    def __init__(
        self, channels: int, n_flows: int = 15, wn_layers: int = 12, wn_width: int = 32
    ):
        super(WaveGlow, self).__init__()
        self.params = clean_init_args(locals().copy())
        self.channels = channels
        self.n_flows = n_flows

        # This will be a channel-wise time-global scaling parameter
        self.a = nn.Parameter(torch.ones(1, self.channels, 1), requires_grad=True)
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
        bs, _, l = m.shape
        f_m = torch.zeros((bs, self.channels, l), device=m.device, dtype=m.dtype)
        f_m[:, 0, :] = m[:, 0, :]

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
            self.ℒ.__setattr__(f'skip_{k}', F.mse_loss(f_m, self.S))

            # Save for loss
            ℒ_det_W += log_det_W
            ℒ_log_s += log_s.mean()

        # Apply final global scaler
        S_tilde = self.a * f_m

        self.ℒ.det_W = -ℒ_det_W
        self.ℒ.log_s = - ℒ_log_s

        return S_tilde

    def infer(self, m: torch.Tensor) -> torch.Tensor:
        return self.forward(m)

    def test(self, m: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        σ = 1.
        α, β = 1., 1.
        self.S = S
        S_tilde = self.forward(m)
        self.ℒ.p_z_likelihood = α * (S_tilde ** 2).mean() / (2 * σ ** 2)
        self.ℒ.reconstruction = β * F.mse_loss(S_tilde, S)
        ℒ = self.ℒ.p_z_likelihood + self.ℒ.det_W + self.ℒ.log_s + self.ℒ.reconstruction
        for k in range(self.n_flows):
            ℒ = ℒ + self.ℒ.__getattribute__(f'skip_{k}')
        return ℒ
