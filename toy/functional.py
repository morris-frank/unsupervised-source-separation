import random
import torch


def destroy_along_axis(x: torch.Tensor, amount: float) -> torch.Tensor:
    """
    Destroys a random subsample of channels in a Tensor ([N,C,L]).
    Destroy == set to 0.

    Args:
        x: Input tensor
        amount: percentage amount to destroy

    Returns:
        Destroyed x
    """
    if amount == 0:
        return x
    length = x.shape[1]
    for i in random.sample(range(length), int(amount * length)):
        if x.ndim == 3:
            x[:, i, :] = 0.
        else:
            x[:, i] = 0.
    return x


def toy2argmax(y_tilde: int, ns: int, μ: int = 101):
    """
    Takes argmax from SoftMax-output over the concatenated channels.

    Args:
        y_tilde: Output of network pred
        ns: Number of sources
        μ: Number of classes for μ-law encoding

    Returns:
        Argmaxed y_tilde with only ns channels
    """
    assert y_tilde.shape[1] == ns * μ
    signals = []
    for i in range(ns):
        j = i * μ
        signals.append(y_tilde[:, j:j + μ, :].argmax(dim=1))
    return torch.cat(signals)
