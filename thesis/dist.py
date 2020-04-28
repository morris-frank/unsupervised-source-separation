from math import pi as π
from math import sqrt, log

import torch
from torch import distributions as dist
from torch.distributions import constraints
from torch import Tensor as T


class AffineBeta(dist.Beta):
    def __init__(
        self, *args, s: float = 2.0, t: float = -1.0, ε: float = 1e-4, **kwargs
    ):
        super(AffineBeta, self).__init__(*args, **kwargs)
        self.support = constraints.interval(t, s + t)
        self.s, self.t, self.ε = s, t, ε

    @property
    def α(self) -> float:
        return self.concentration1

    @property
    def β(self) -> float:
        return self.concentration0

    @property
    def mean(self) -> T:
        return self.s * super(AffineBeta, self).mean + self.t

    def rsample(self, sample_shape=()) -> T:
        samp = self.s * super(AffineBeta, self).rsample(sample_shape) + self.t
        return samp.clamp(self.t + self.ε, self.s + self.t - self.ε)

    def log_prob(self, samples: T) -> T:
        samples = (samples - self.t) / self.s
        return super(AffineBeta, self).log_prob(samples)


def norm_cdf(samples: T) -> T:
    """
    Element-wise cumulative value for tensor x under a normal distribution.

    Args:
        samples: the tensor

    Returns:
        the norm cdfs
    """
    return (1. + torch.erf(samples / sqrt(2.))) / 2.


def norm_log_prob(samples: T, μ: T, log_σ: T) -> T:
    return (
        -0.5 * log(2.0 * π) - log_σ - 0.5 * (samples - μ) ** 2 / torch.exp(2.0 * log_σ)
    )


def norm_truncate_rsample(
    μ: T, σ: T, ll: bool = False, a: float = -1.0, b: float = 1.0
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
