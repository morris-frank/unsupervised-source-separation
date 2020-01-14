import random
import torch


def destroy_along_axis(x: torch.Tensor, amount: float) -> torch.Tensor:
    if amount == 0:
        return x
    length = x.shape[1]
    for i in random.sample(range(length), int(amount * length)):
        if x.ndim == 3:
            x[:, i, :] = 0.
        else:
            x[:, i] = 0.
    return x


def toy2argmax(logits, ns):
    μ = 101
    signals = []
    for i in range(ns):
        j = i * μ
        signals.append(logits[:, j:j + μ, :].argmax(dim=1))
    return torch.cat(signals)
