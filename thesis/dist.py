from math import pi as π
from math import sqrt, log

import torch
from torch import distributions as dist


def norm_cdf(x: torch.Tensor):
    """
    Element-wise cumulative value for tensor x under a normal distribution.

    Args:
        x: the tensor

    Returns:
        the norm cdfs
    """
    return (1.0 + torch.erf(x / sqrt(2.0))) / 2.0


def rsample_truncated_normal(
    μ: torch.Tensor, σ: torch.Tensor, ll: bool = False, a: float = -1.0, b: float = 1.0
):
    """
    Takes an rsample from a truncated normal distribution given the mean μ and
    the variance σ. Sample is same sized as μ and σ.

    Args:
        μ: Means
        σ: Variances
        ll: if true also returns the log likelihood of the samples!
        a: Left/lower bound/truncation
        b: Right/upper bound/truncation

    Returns:
        A sample from the truncated normal sized as μ/σ.
    """
    assert μ.shape == σ.shape

    l = norm_cdf((-μ + a) / σ)
    u = norm_cdf((-μ + b) / σ)

    udist = dist.uniform.Uniform(2 * l - 1, 2 * u - 1)
    tensor = udist.rsample()

    tensor.erfinv_()
    tensor.mul_(σ * sqrt(2.0))
    tensor.add_(μ)

    tensor.clamp_(a, b)

    if ll:
        denom = torch.log(u - l)
        ξ = (tensor - μ) / σ
        num = -0.5 * ξ * ξ - 0.5 * log(2.0 * π)
        log_l = -torch.log(σ) + num - denom
        return tensor, log_l
    else:
        return tensor


def likelihood_truncated_normal(
    x: torch.Tensor, μ: torch.Tensor, σ: torch.Tensor, a: float = -1.0, b: float = 1.0
):
    assert μ.shape == σ.shape

    l = norm_cdf((-μ + a) / σ)
    u = norm_cdf((-μ + b) / σ)

    ξ = (x - μ) / σ
    φ_ξ = 1 / sqrt(2 * π) * torch.exp(-0.5 * ξ * ξ)

    f_x = φ_ξ / (u - l) / σ

    return f_x


def likelihood_normal(x, μ, log_σ):
    return -0.5 * log(2.0 * π) - log_σ - 0.5 * (x - μ) ** 2 / torch.exp(2.0 * log_σ)
