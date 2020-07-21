import torch
from math import sqrt


def langevin_sample(model, σ, m):
    N, C, L = m.shape
    η = .00003 * (σ / .01)**2
    λ = 1./σ**2
    # λ = 0

    with torch.enable_grad():
        model.train()
        ŝ = (0.1 * torch.randn((N, 4*C, L), device=m.device)).clamp(-1, 1)
        ŝ.requires_grad = True
        for i in range(100):
            yield ŝ
            model.zero_grad()
            _, log_p, _ = model(ŝ)
            ℒ = -log_p.clamp(-1e5, 1e5).mean()
            ℒ.backward()
            δŝ = ŝ.grad
            ŝ.detach_()
            ε = sqrt(2 * η) * torch.randn_like(ŝ)
            m_ = torch.stack(ŝ.chunk(4, 1), 0).sum(0) - m
            ŝ = ŝ + η * (δŝ - λ * m_.repeat(1, 4, 1)) + ε
            ŝ.requires_grad = True
        return ŝ
