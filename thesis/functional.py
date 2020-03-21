import random
from math import log, sqrt
from math import pi as π
from typing import Tuple

import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm


def remove_list_weight_norm(ml: nn.ModuleList) -> nn.ModuleList:
    """
    Removes weight norm from all layers in a ModuleList

    Args:
        ml: the ModuleList

    Returns:
        the changes modulelist
    """
    _ml = nn.ModuleList([remove_weight_norm(l) for l in ml])
    return _ml


# @torch.jit.script
def dilate(x: torch.Tensor, new: int, old: int = 1) -> torch.Tensor:
    """
    Will dilate the input tensor of shape [N, C, L]

    Args:
        x: input tensor
        new: new amount of dilation to get
        old: current amount if dilation of tensor

    Returns:
        dilated tensor
    """
    n, c, w = x.shape  # N == Batch size × old
    dilation = new / old
    if dilation == 1:
        return x
    w, n = int(w / dilation), int(n * dilation)
    x = x.permute(1, 2, 0)
    x = torch.reshape(x, [c, w, n])
    x = x.permute(2, 0, 1)
    return x.contiguous()


def shift1d(x: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Shifts a Tensor to the left or right and pads with zeros.

    Args:
        x: Input Tensor [N×C×L]
        shift: the shift, negative for left shift, positive for right

    Returns:
        the shifted tensor, same size
    """
    assert x.ndimension() == 3
    length = x.shape[2]
    pad = [-min(shift, 0), max(shift, 0)]
    y = F.pad(x, pad)
    y = y[:, :, pad[1] : pad[1] + length]
    return y.contiguous()


def encode_μ_law(x: torch.Tensor, μ: int = 255) -> torch.Tensor:
    """
    Encodes the input tensor element-wise with μ-law encoding

    Args:
        x: tensor
        μ: the size of the encoding (number of possible classes)

    Returns:
        the encoded tensor
    """
    assert μ & 1
    assert x.max() <= 1.0 and x.min() >= -1.0
    μ = μ - 1
    out = torch.sign(x) * torch.log(1 + μ * torch.abs(x)) / log(μ)
    out = torch.round(out * (μ // 2))
    return out


def decode_μ_law(x: torch.Tensor, μ: int = 255) -> torch.Tensor:
    """
    Applies the element-wise inverse μ-law encoding to the tensor.

    Args:
        x: input tensor
        μ: size of the encoding (number of possible classes)

    Returns:
        the decoded tensor
    """
    assert μ & 1
    x = x.type(torch.float32)
    # out = (tensor + 0.5) * 2. / (μ + 1)
    μ = μ - 1
    out = x / (μ // 2)
    out = torch.sign(out) / μ * (torch.pow(μ, torch.abs(out)) - 1)
    return out


def destroy_along_channels(x: torch.Tensor, amount: float) -> torch.Tensor:
    """
    Destroys a random subsample of channels in a Tensor ([N,C,L]).
    Destroy == set to 0.

    Args:
        x: Input tensor
        amount: percentage amount to destroy

    Returns:
        Destroyed tensor
    """
    if amount == 0:
        return x
    length = x.shape[1]
    for i in random.sample(range(length), int(amount * length)):
        if x.ndim == 3:
            x[:, i, :] = 0.0
        else:
            x[:, i] = 0.0
    return x


def multi_argmax(x: torch.Tensor, n: int, μ: int = 101) -> torch.Tensor:
    """
    Takes argmax from SoftMax-output over the concatenated channels.

    Args:
        x: Output of network pred
        n: Number of sources
        μ: Number of classes for μ-law encoding

    Returns:
        Argmaxed y_tilde with only ns channels
    """
    assert x.shape[1] == n * μ
    signals = []
    for i in range(n):
        j = i * μ
        signals.append(x[:, j : j + μ, :].argmax(dim=1))
    return torch.cat(signals)


def orthonormal(*dim: int) -> torch.Tensor:
    """
    Creates an orthonormal Tensor with the given dimensions.

    Args:
        *dim: The dimensions.

    Returns:
        the Tensor (new)
    """
    # this guarantees |det(weights)| == 1 but not the sign
    data = torch.qr(torch.empty(*dim).normal_())[0]

    if torch.det(data) < 0:
        data[:, 0] = -data[:, 0]
    return data


def split(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return x[:, :, ::2].contiguous(), x[:, :, 1::2].contiguous()


def interleave(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    assert left.shape == right.shape
    bs, c, l = left.shape
    return torch.stack((left, right), dim=3).view(bs, c, 2 * l)


def split_LtoC(x: torch.Tensor) -> torch.Tensor:
    N, C, L = x.shape
    squeezed = x.view(N, C, L // 2, 2).permute(0, 1, 3, 2)
    out = squeezed.contiguous().view(N, C * 2, L // 2)
    return out


def flip_channels(x: torch.Tensor) -> torch.Tensor:
    x_a, x_b = x.chunk(2, 1)
    return torch.cat([x_b, x_a], 1)


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
        num = -0.5 * ξ * ξ - 0.5 * log(2 * π)
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
    return -0.5 * log(2. * π) - log_σ - 0.5 * (x - μ) ** 2 / torch.exp(2. * log_σ)
