from torch.nn import functional as F
import torch
from torch import nn

from . import BaseModel
from ..optim import sqrt_l1_loss
from ..wavenet import Wavenet
from ...functional import split, interleave
from ...utils import clean_init_args


class RealNVP(BaseModel):
    def __init__(
        self, channels: int, n_flows: int = 15, wn_layers: int = 12, wn_width: int = 32
    ):
        super(RealNVP, self).__init__()
        self.params = clean_init_args(locals().copy())

        self.channels = channels
        self.n_flows = n_flows

        # This will be a channel-wise time-global scaling parameter
        self.a = nn.Parameter(torch.ones(1, self.channels, 1), requires_grad=True)
        self.waves = nn.ModuleList()
        for _ in range(n_flows):
            self.waves.append(
                Wavenet(
                    in_channels=channels,
                    out_channels=channels * 2,
                    n_blocks=1,
                    n_layers=wn_layers,
                    residual_width=2 * wn_width,
                    skip_width=wn_width,
                    zero_final=True
                )
            )

    def apply_wave(self, x: torch.Tensor, k: int):
        x_left, x_right = split(x)

        if k % 2 == 0:
            log_s_t = self.waves[k](x_left)
        else:
            log_s_t = self.waves[k](x_right)

        log_s, t = log_s_t.chunk(2, dim=1)
        return x_left, x_right, log_s, t

    def forward(self, m: torch.Tensor, *args):
        bs, _, l = m.shape
        f_m = torch.zeros((bs, self.channels, l), device=m.device, dtype=m.dtype)
        f_m[:, 0, :] = m[:, 0, :]

        ℒ_log_s = 0
        for k in range(self.n_flows):
            # Separate them and get scale and translate
            m_a, m_b, log_s, t = self.apply_wave(f_m, k)

            if k % 2 == 0:
                m_b = log_s.exp() * m_b + t
            else:
                m_a = log_s.exp() * m_a + t

            f_m = interleave(m_a, m_b)

            # Sum log_s over batch size
            #ℒ_log_s = log_s.view(log_s.size(0), -1).sum(-1) + ℒ_log_s
            ℒ_log_s += log_s.mean()

        S_tilde = self.a * f_m
        self.ℒ.log_s = -ℒ_log_s
        return S_tilde

    def infer(self, z: torch.Tensor):
        f_z = z

        for k in reversed(range(self.n_flows)):
            z_a, z_b, log_s, t = self.apply_wave(f_z, k)

            if k % 2 == 0:
                z_b = (z_b - t) / log_s.exp()
            else:
                z_a = (z_a - t) / log_s.exp()

            f_z = interleave(z_a, z_b)

        m = f_z[:, 0, :]
        return m

    def test(self, m: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        σ = 1.0
        α, β = 1.0, 1.0
        S_tilde = self.forward(m)
        self.ℒ.p_z_likelihood = α * (S_tilde ** 2).mean() / (2 * σ ** 2)
        self.ℒ.reconstruction = β * F.l1_loss(S_tilde, S)
        #self.ℒ.reconstruction = β * sqrt_l1_loss(S_tilde, S)
        #ℒ = self.ℒ.p_z_likelihood + self.ℒ.reconstruction + self.ℒ.log_s
        ℒ = self.ℒ.reconstruction
        return ℒ
