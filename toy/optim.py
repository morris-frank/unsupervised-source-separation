from itertools import permutations
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def toy_loss_ordered(ns: int, μ: int = 101):
    def loss_function(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                      device: str) -> Tuple[None, torch.Tensor]:
        logits = model(x)
        loss = None
        for i in range(ns):
            _loss = F.cross_entropy(logits[:, i * μ:i * μ + μ, :],
                                    y[:, i, :].to(device))
            loss = loss + _loss if loss else _loss
        return None, loss

    return loss_function


def toy_loss_unordered(ns: int, μ: int = 101):
    def loss_function(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                      device: str) -> Tuple[None, torch.Tensor]:
        logits = model(x)

        loss = None
        # We have to go through ALL possible assignments
        for ordering in permutations(range(ns)):
            order_loss = None
            for i in range(ns):
                j = ordering[i]
                _loss = F.cross_entropy(logits[:, j * μ:j * μ + μ, :],
                                        y[:, i, :].to(device))
                order_loss = order_loss + _loss if order_loss else _loss
            loss = order_loss if (not loss or order_loss < loss) else loss
        return None, loss

    return loss_function
