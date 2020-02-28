import torch

from .vqvae import VQVAE as _VQVAE
from .nvp import RealNVP
from .waveglow import WaveGlow
from ...utils import clean_init_args


class PriorNVP(RealNVP):
    def __init__(self, k: int, *args, **kwargs):
        super(PriorNVP, self).__init__(channels=1, *args, **kwargs)
        self.params = clean_init_args(locals().copy())

        del self.a
        self.a = 1.
        self.name = k
        self.k = k

    def test(self, s: torch.Tensor, *args) -> torch.Tensor:
        α, σ = 1., 1.
        z = self.forward(s)
        self.ℒ.p_z_likelihood = α * (z*z).mean() / (2*σ*σ)
        ℒ = self.ℒ.p_z_likelihood + self.ℒ.log_s
        return ℒ.mean()


class PriorGlow(WaveGlow):
    def __init__(self, k: int, channels: int, *args, **kwargs):
        super(PriorGlow, self).__init__(channels, *args, **kwargs)

        self.name = k
        self.k = k

    def test(self, s: torch.Tensor, *args) -> torch.Tensor:
        α, σ = 1., 1.
        z = self.forward(s)
        self.ℒ.p_z_likelihood = α * (z*z).mean() / (2*σ*σ)
        ℒ = self.ℒ.p_z_likelihood + self.ℒ.log_s + self.ℒ.det_W
        return ℒ.mean()


class VQVAE(_VQVAE):
    def __init__(self, k: int, *args, **kwargs):
        super(VQVAE, self).__init__(*args, **kwargs)
        self.name = k
        self.k = k
