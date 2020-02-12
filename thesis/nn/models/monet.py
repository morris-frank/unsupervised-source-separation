from typing import Optional
import torch
from torch.nn import functional as F
from torch import nn

from .attention import Attention
from .compvae import ComponentVAE
from ...utils import clean_init_args


class MONet(nn.Module):
    def __init__(self, slots: int):
        super(MONet, self).__init__()
        self.params = clean_init_args(locals().copy())
        self.slots = slots
        self.attention = Attention(in_channels=1, out_channels=1, ngf=32)
        self.component = ComponentVAE(in_channels=1)
        self.ε = torch.finfo(torch.float).eps

        # Save losses during training:
        self.ℒ_E = None  # Encoder loss
        self.ℒ_D = None  # Decoder loss
        self.ℒ_R = None  # Possible reconstruction loss
        self.losses = dict(encoder=[], decoder=[], reconstruction=[], mask=[])

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None):
        bs, _, h, w = x.shape

        if S is not None:
            assert S.shape[1] == self.slots

        log_s_k = x.new_zeros((bs, 1, h, w))
        x_tilde = 0

        self.ℒ_E = 0
        self.ℒ_R = 0
        b = []
        m = []
        m_tilde_logits = []

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
            self.ℒ_E += -0.5 * (1 + z_logvar_k - z_μ_k.pow(2) - z_logvar_k.exp()).sum()

            m_k = log_m_k.exp()
            x_k_masked = m_k * x_μ_k

            if S is not None:
                self.ℒ_R += F.mse_loss(x_k_masked, S[:, k, ...])

            # Exponents for the decoder loss
            b_k = (
                log_m_k - 0.5 * x_logvar_k - (x - x_μ_k).pow(2) / (2 * x_logvar_k.exp())
            )
            b.append(b_k.unsqueeze(1))

            # Iteratively reconstruct the output image
            x_tilde += x_k_masked
            # Accumulate
            m.append(m_k)
            m_tilde_logits.append(m_tilde_k_logits)

        b = torch.cat(b, dim=1)
        m = torch.cat(m, dim=1)
        m_tilde_logits = torch.cat(m_tilde_logits, dim=1)

        self.ℒ_E /= bs
        self.ℒ_D = -torch.logsumexp(b, dim=1).sum() / bs

        return x_tilde, m, m_tilde_logits

    def loss(self, β: float = 0.5, γ: float = 0.5):
        def func(model, x, y, progress):
            _, m, m_tilde_logits = model(x, y)
            ℒ_M = F.kl_div(m_tilde_logits.log_softmax(dim=1), m, reduction="batchmean")
            ℒ = self.ℒ_D + β * self.ℒ_E + γ * ℒ_M + self.ℒ_R

            self.losses['encoder'].append(self.ℒ_E.detach().item())
            self.losses['decoder'].append(self.ℒ_D.detach.item())
            self.losses['reconstruction'].append(self.ℒ_R.detach().item())
            self.losses['mask'].append(ℒ_M.detach().item())
            return ℒ

        return func