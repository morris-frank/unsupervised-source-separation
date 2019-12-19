import torch


def toy2argmax(logits, ns):
    μ = 101
    signals = []
    for i in range(ns):
        j = i * μ
        signals.append(logits[:, j:j + μ, :].argmax(dim=1))
    return torch.cat(signals)
