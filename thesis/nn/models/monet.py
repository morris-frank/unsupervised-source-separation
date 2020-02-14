from typing import Optional

import torch
from torch.nn import functional as F

from . import BaseModel
from ..attention import Attention
from ..compvae import ComponentVAE
from ...utils import clean_init_args


class MONet(BaseModel):
    def __init__(self, slots: int):
        super(MONet, self).__init__()
        self.params = clean_init_args(locals().copy())
        self.slots = slots
        self.attention = Attention(in_channels=1, out_channels=1, ngf=64)
        self.component = ComponentVAE(in_channels=1)
        self.ε = torch.finfo(torch.float).eps

    def forward(
        self, x: torch.Tensor, S: Optional[torch.Tensor] = None, infer: bool = False
    ):
        bs, _, h, w = x.shape

        if S is not None:
            assert S.shape[1] == self.slots

        log_s_k = x.new_zeros((bs, 1, h, w))

        ℒ_encoder = 0
        ℒ_recon = 0
        b = []
        m = []
        m_tilde_logits = []
        x_tilde = [] if infer else 0

        for k in range(self.slots):
            if k != self.slots - 1:
                log_α_k = self.attention(x, log_s_k)
                log_m_k = log_s_k + log_α_k
                log_s_k += (1.0 - log_α_k.exp()).clamp(min=self.ε).log()
            else:
                log_m_k = log_s_k

            m_tilde_k_logits, x_μ_k, x_logvar_k, z_μ_k, z_logvar_k = self.component(
                x, log_m_k, k == 0
            )

            # KLD is additive for independent distributions
            ℒ_encoder += -0.5 * (1 + z_logvar_k - z_μ_k.pow(2) - z_logvar_k.exp()).sum()

            m_k = log_m_k.exp()
            x_k_masked = m_k * x_μ_k
            if infer:
                x_tilde.append(x_k_masked.detach())
            else:
                x_tilde += x_k_masked

            if S is not None:
                ℒ_recon += F.mse_loss(x_k_masked, S[:, k, ...])

            # Exponents for the decoder loss
            b_k = (
                log_m_k - 0.5 * x_logvar_k - (x - x_μ_k).pow(2) / (2 * x_logvar_k.exp())
            )
            b.append(b_k.unsqueeze(1))

            # Accumulate
            m.append(m_k)
            m_tilde_logits.append(m_tilde_k_logits)

        b = torch.cat(b, dim=1)
        m = torch.cat(m, dim=1)
        m_tilde_logits = torch.cat(m_tilde_logits, dim=1)

        self.ℒ.encoder = ℒ_encoder / bs
        self.ℒ.decoder = -torch.logsumexp(b, dim=1).sum() / bs
        self.ℒ.recon = ℒ_recon

        if infer:
            return torch.cat(x_tilde, dim=1)
        return x_tilde, m, m_tilde_logits

    def infer(self, x):
        x_tilde = self.forward(x, infer=True)
        return x_tilde

    def test(self, m: torch.Tensor, S: Optional[torch.Tensor]) -> torch.Tensor:
        β, γ, ρ = 0.5, 0.5, 1.0
        _, m, m_tilde_logits = self.forward(m, S)
        self.ℒ.mask = F.kl_div(
            m_tilde_logits.log_softmax(dim=1), m, reduction="batchmean"
        )

        ℒ = self.ℒ.decoder + β * self.ℒ.encoder + γ * self.ℒ.mask + ρ * self.ℒ.recon
        return ℒ
