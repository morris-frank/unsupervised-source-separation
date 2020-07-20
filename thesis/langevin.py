import torch
from math import sqrt
from matplotlib import pyplot as plt


def langevin_sample(model, σ, m):
    N, C, L = m.shape
    η = .00003 * (σ / .01)**2
    λ = 1./σ**2
    λ = 0

    model.train()
    ŝ = torch.randn((N, 4*C, L), requires_grad=True)
    for i in range(100):
        print(f"Step {i}", end='\t')
        for k in range(4):
            plt.imshow(ŝ[0, k*80:(k+1)*80, 1000:1500].detach().numpy())
            plt.savefig(f"/home/morris/ŝ_{k}_{i}.png")
            plt.close()
        model.zero_grad()
        _, log_p, _ = model(ŝ)
        ℒ = -log_p.clamp(-1e5, 1e5).mean()
        print(f"mean ℒ: {ℒ}", end="\t")
        ℒ.backward()
        δŝ = ŝ.grad
        print(f"mean δ: {δŝ.mean()}", end="\t")
        ŝ.detach_()
        ε = sqrt(2 * η) * torch.randn_like(ŝ)
        m_ = torch.stack(ŝ.chunk(4, 1), 0).sum(0) - m
        print(f"mean constraint: {m_.mean()}")
        ŝ = ŝ + η * (δŝ - λ * m_.repeat(1, 4, 1)) + ε
        ŝ.requires_grad = True
    return ŝ
