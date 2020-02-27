import torch

from .nvp import RealNVP
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
        σ = 1.0
        α = 1.0
        z = self.forward(s)
        self.ℒ.p_z_likelihood = α * (z ** 2).mean() / (2 * σ ** 2)
        ℒ = self.ℒ.p_z_likelihood + self.ℒ.log_s
        return ℒ
