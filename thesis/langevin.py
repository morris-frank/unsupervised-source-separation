import torch
from math import sqrt
from torch import autograd


def langevin_sample(model, σ, m, ŝ=None):
    N, C, L = m.shape
    η = .00003 * (σ / .01)**2
    λ = 1./σ**2
    λ = 0

    δŝ = torch.zeros_like(ŝ)

    # s = torch.zeros_like(ŝ)
    # s[:, 0, :] = ŝ[:, -1, :]
    # s[:, -1, :] = ŝ[:, 0, :]
    # s[:, 1, :] = ŝ[:, 2, :]
    # s[:, 2, :] = ŝ[:, 1, :]
    # ŝ = s
    _, L, _ = model(ŝ)
    yield ŝ, -L.mean().detach().item(), δŝ
    ŝ = (ŝ + 0.4 * torch.randn_like(ŝ)).clamp(-1, 1)

    with torch.enable_grad():
        if ŝ is None:
            ŝ = (0.1 * torch.randn((N, 4*C, L), device=m.device)).clamp(-1, 1)
        ŝ.requires_grad = True
        ℒ = 0
        for i in range(300):
            yield ŝ, ℒ, δŝ
            _, ℒ, _ = model(ŝ)
            δŝ = autograd.grad((-ℒ.mean()), ŝ, only_inputs=True)[0]
            ε = sqrt(2 * η) * torch.randn_like(ŝ)
            m_ = (torch.stack(ŝ.chunk(4, 1), 0).sum(0) - m).repeat(1, 4, 1)
            ŝ = ŝ.detach().add(η * (δŝ - λ * m_) + 0.05 * ε).clamp(-1, 1)
            ℒ = list(map(lambda x: f"{x:.3}", ℒ.detach().cpu().mean(-1).squeeze().tolist()))
    return ŝ, ℒ, δŝ
