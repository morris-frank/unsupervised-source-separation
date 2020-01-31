import torch
from torch import nn
from torch.autograd import Variable

from typing import Tuple

from .wavenet import Wavenet
from ...utils import clean_init_args


class RealNVP(nn.Module):
    def __init__(self, channels: int, n_flows: int = 10, wn_layers: int = 12,
                 wn_width: int = 32):
        super(RealNVP, self).__init__()
        self.params = clean_init_args(locals().copy())

        self.channels = channels
        self.n_flows = n_flows
        self.window_size = 2**9

        self.waves = nn.ModuleList()
        for _ in range(n_flows):
            self.waves.append(Wavenet(
                in_channels=channels, out_channels=channels * 2,
                c_channels=1, n_blocks=1, n_layers=wn_layers,
                residual_width=2 * wn_width, skip_width=wn_width)
            )

    def _split_and_flow(self, x: torch.Tensor, k: int, c: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        windows = torch.split(x, self.window_size, dim=2)
        left = torch.cat(windows[::2], dim=2)
        right = torch.cat(windows[1::2], dim=2)

        if k % 2 == 0:
            log_s_t = self.waves[k](left, c)
        else:
            log_s_t = self.waves[k](right, c)

        log_s, t = log_s_t.chunk(2, dim=1)
        return left, right, log_s, t

    def _interleave_windows(self, left: torch.Tensor, right: torch.Tensor) \
            -> torch.Tensor:
        left = left.split(self.window_size, dim=2)
        right = right.split(self.window_size, dim=2)

        # Interleave the masks and combine back together
        windows = [win for pair in zip(left, right) for win in pair]
        return torch.cat(windows, dim=2)

    def forward(self, m: torch.Tensor, S: torch.Tensor):
        assert S.shape[1] == self.channels

        f_s = S
        total_log_s = 0
        for k in range(self.n_flows):
            # Separate them and get scale and translate
            s_a, s_b, log_s, t = self._split_and_flow(f_s, k, m)

            if k % 2 == 0:
                s_b = log_s.exp() * s_a + t
            else:
                s_a = log_s.exp() * s_b + t

            f_s = self._interleave_windows(s_a, s_b)

            # Save for loss
            total_log_s = log_s.sum() + total_log_s
        z = f_s
        return z, total_log_s

    def infer(self, m: torch.Tensor, σ: float = 1.):
        N, _, L = m.shape

        # Sample a z
        f_z = torch.empty(N, self.channels, L).type(m.dtype).to(m.device)
        f_z = Variable(σ * f_z.normal_())

        for k in reversed(range(self.n_flows)):
            z_a, z_b, log_s, t = self._split_and_flow(f_z, k, m)

            if k % 2 == 0:
                z_b = (z_a - t) / log_s.exp()
            else:
                z_a = (z_b - t) / log_s.exp()

            f_z = self._interleave_windows(z_a, z_b)

        return f_z

    @staticmethod
    def loss(σ: float = 1.):
        def func(model, x, y, progress):
            _ = progress
            z, total_log_s = model(x, y)
            loss = (z * z).sum() / (2 * σ ** 2) - total_log_s
            return loss / z.numel()

        return func
