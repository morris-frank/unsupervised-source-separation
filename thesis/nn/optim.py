import torch
from torch import distributions as dist
from torch.nn import functional as F


def multi_cross_entropy(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Get cross entropy of a multiple multi-class problem.

    Args:
        x: Probabilities for all samples and classes size
        t: Correct indices for all samples size n

    Returns:
        The sum of n CE losses
    """
    assert x.shape[1] == t.shape[1]
    loss = None
    for i in range(x.shape[1]):
        _x = x[:, i, :, :]
        _t = t[:, i, :].to(x.device)
        _loss = F.cross_entropy(_x, _t)
        loss = loss + _loss if loss else _loss
    return loss / x.shape[1]


def sample_kl(x_q: torch.Tensor, x_q_log_prob: torch.Tensor) -> torch.Tensor:
    """
    Takes a sample of the Kullback-Leibler divergence between given input
    and a same sized standard gaussian.
    Args:
        x_q:
        x_q_log_prob:

    Returns:
        The sample value of KL
    """
    zx_p_loc = torch.zeros_like(x_q).to(x_q.device)
    zx_p_scale = torch.ones_like(x_q).to(x_q.device)
    pzx = dist.Normal(zx_p_loc, zx_p_scale)
    kl_zx = torch.sum(pzx.log_prob(x_q) - x_q_log_prob)
    return kl_zx
