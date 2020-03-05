from typing import List, Tuple

import torch
from torch.nn import functional as F

from . import BaseModel
from ..old_wavenet import Wavenet
from ...functional import rsample_truncated_normal
from ...utils import clean_init_args


class UMixer(BaseModel):
    def __init__(self, priors: List):
        super(UMixer, self).__init__()
        self.params = clean_init_args(locals().copy())

        self.n_classes = len(priors)

        self.q_sǀm = Wavenet()

        self.p_s = priors

        self.p_mǀs = Wavenet()

    def forward(self, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        η = self.q_sǀm(m)
        μ, σ = η.chunk(dim=1)

        # yes?
        μ = torch.sigmoid(μ)
        σ = F.softplus(σ) + 1e-7

        s_, log_q_s_ = rsample_truncated_normal(μ, σ)

        for k in range(self.n_classes):
            # Get Log likelihood under prior
            log_p_s_, _ = self.p_s[k](s_[:, None, k, :])

            # Kullback Leibler for this k'th source
            KL_k = torch.sum(log_p_s_ - log_q_s_)
            self.ℒ.__setattr__(f"KL_{k}", KL_k)

        m_ = self.p_mǀs(s_)
        self.ℒ.l1_recon = F.l1_loss(m_, m)

        return s_, m_

    def test(self, m: torch.Tensor) -> torch.Tensor:
        β = 1.1
        _ = self.forward(m)

        ℒ = self.ℒ.l1_recon

        for k in range(self.n_classes):
            ℒ -= β * self.ℒ.__getattr__(f"KL_{k}")

        return ℒ
