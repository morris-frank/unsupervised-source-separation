from typing import Tuple, Optional

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from . import BaseModel
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
        self.window_size = 2 ** 9

        self.waves = nn.ModuleList()
        for _ in range(n_flows):
            self.waves.append(
                Wavenet(
                    in_channels=channels,
                    out_channels=channels * 2,
                    c_channels=1,
                    n_blocks=1,
                    n_layers=wn_layers,
                    residual_width=2 * wn_width,
                    skip_width=wn_width,
                )
            )

    def apply_wave(
        self, x: torch.Tensor, k: int, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_left, x_right = split(x)
        c_left, c_right = split(c)

        if k % 2 == 0:
            log_s_t = self.waves[k](x_left, c_left)
        else:
            log_s_t = self.waves[k](x_right, c_right)

        log_s, t = log_s_t.chunk(2, dim=1)
        return x_left, x_right, log_s, t

    def forward(self, m: torch.Tensor, S: torch.Tensor):
        assert S.shape[1] == self.channels

        f_s = S
        ℒ_log_s = 0
        for k in range(self.n_flows):
            # Separate them and get scale and translate
            s_a, s_b, log_s, t = self.apply_wave(f_s, k, m)

            if k % 2 == 0:
                s_b = log_s.exp() * s_b + t
            else:
                s_a = log_s.exp() * s_a + t

            f_s = interleave(s_a, s_b)

            # Save for loss
            ℒ_log_s += log_s.sum()
        z = f_s
        self.ℒ.log_s = ℒ_log_s
        return z

    def infer(
        self,
        m: torch.Tensor,
        σ: float = 1.0,
        μ: float = 0.0,
        z: Optional[torch.Tensor] = None,
    ):
        N, _, L = m.shape

        # Sample a z
        if z is None:
            z = torch.empty(N, self.channels, L)
            z.normal_()
        f_z = Variable(σ * z.type(m.dtype).to(m.device) + μ)

        for k in reversed(range(self.n_flows)):
            z_a, z_b, log_s, t = self.apply_wave(f_z, k, m)

            if k % 2 == 0:
                z_b = (z_b - t) / log_s.exp()
            else:
                z_a = (z_a - t) / log_s.exp()

            f_z = interleave(z_a, z_b)

        return f_z

    def loss(self, m: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        σ = 1.0
        z = self(m, S)
        self.ℒ.z = (z * z).sum() / (2 * σ ** 2)
        ℒ = self.ℒ.z - self.ℒ.log_s
        ℒ /= z.numel()
        return ℒ


class ConditionalRealNVP(RealNVP):
    def __init__(self, classes, *args, **kwargs):
        super(ConditionalRealNVP, self).__init__(channels=1, *args, **kwargs)
        self.params = clean_init_args(locals().copy())
        self.classes = classes

        self.conditioner = nn.Sequential(nn.Linear(classes, 32), nn.Linear(32, 2))

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], s: torch.Tensor):
        m, y = x  # y is channel label
        y = F.one_hot(y, self.classes).float().to(m.device)
        σ, μ = self.conditioner(y).unsqueeze(-1).chunk(2, dim=1)
        σ = F.softplus(σ) + 1e-7
        f_s = super(ConditionalRealNVP, self).forward(m, s)
        z = (f_s - μ) / σ
        return z, σ

    def infer(
        self, x: Tuple[torch.Tensor, torch.Tensor], z: Optional[torch.Tensor] = None
    ):
        m, y = x  # y is channel label
        y = F.one_hot(y, self.classes).float().to(m.device)
        σ, μ = self.conditioner(y).unsqueeze(-1).chunk(2, dim=1)
        σ = F.softplus(σ) + 1e-7
        f_z = super(ConditionalRealNVP, self).infer(m, σ, μ, z)
        return f_z

    def loss(self, m: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        z, σ = self(m, S)
        self.ℒ.z = ((z * z) / (2 * σ ** 2)).sum()
        ℒ = self.ℒ.z - self.ℒ.log_s
        ℒ /= z.numel()
        return ℒ
