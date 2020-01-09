from typing import Tuple, Callable

import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F


def sum_of_ce(logits: torch.Tensor, y: torch.Tensor, ns: int, μ: int,
              device: str) -> torch.Tensor:
    loss = None
    for i in range(ns):
        _loss = F.cross_entropy(logits[:, i * μ:i * μ + μ, :],
                                y[:, i, :].to(device))
        loss = loss + _loss if loss else _loss
    return loss


def sample_kl(x_q: dist.Normal, x_q_log_prob: torch.Tensor,
              device: str) -> torch.Tensor:
    zx_p_loc = torch.zeros(x_q.size()).to(device)
    zx_p_scale = torch.ones(x_q.size()).to(device)
    pzx = dist.Normal(zx_p_loc, zx_p_scale)
    kl_zx = torch.sum(pzx.log_prob(x_q) - x_q_log_prob)
    return kl_zx


def single_vae_toy_loss(β: float = 1, dβ: float = 1 / 3) -> Callable:
    def loss_function(model: nn.Module, x: Tuple[torch.Tensor, torch.Tensor],
                      y: torch.Tensor, device: str, progress: float) \
            -> Tuple[torch.Tensor, None]:
        mixes, labels = x
        logits, x_q, x_q_log_prob = model(mixes, labels)

        ce_x = F.cross_entropy(logits, y[:, 0, :].to(device))

        kl_zx = sample_kl(x_q, x_q_log_prob, device)

        it_β = min(progress / dβ, 1) * β

        loss = ce_x - it_β * kl_zx

        return loss, None

    return loss_function


def variation_toy_loss_ordered(ns: int, μ: int = 101,
                               dβ: float = 1 / 3, β: float = 1.) -> Callable:
    def loss_function(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                      device: str, progress: float) \
            -> Tuple[torch.Tensor, None]:
        logits, x_q, x_q_log_prob = model(x)

        # First Cross-Entropy
        ce_x = sum_of_ce(logits, y, ns, μ, device)

        # Then Kullback-Leibler
        kl_zx = sample_kl(x_q, x_q_log_prob, device)

        it_β = β * min(progress / dβ, 1)

        loss = ce_x - it_β * kl_zx
        return loss, None

    return loss_function


def toy_loss_ordered(ns: int, μ: int = 101) -> Callable:
    def loss_function(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                      device: str, progress: float) \
            -> Tuple[torch.Tensor, None]:
        del progress
        logits = model(x)
        loss = sum_of_ce(logits, y, ns, μ, device)
        return loss, None

    return loss_function
