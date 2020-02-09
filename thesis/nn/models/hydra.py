from typing import Callable
import torch
from torch import nn
from .wavenet import Wavenet

from ..optim import multi_cross_entropy
from ...functional import multi_argmax
from ...utils import clean_init_args


class Hydra(nn.Module):
    def __init__(self, classes: int, in_channels: int, out_channels: int,
                 wn_width: int):
        super(Hydra, self).__init__()
        self.params = clean_init_args(locals().copy())

        self.classes, self.out_channels = classes, out_channels
        self.bottom = Wavenet(in_channels=in_channels, out_channels=32,
                              residual_width=wn_width * 2, skip_width=wn_width)
        self.heads = nn.ModuleList()
        for _ in range(classes):
            self.heads.append(
                Wavenet(in_channels=32, out_channels=out_channels,
                        residual_width=wn_width * 2, skip_width=wn_width))

    def forward(self, x: torch.Tensor):
        z = self.bottom(x)

        S = [self.heads[k](z) for k in range(self.classes)]
        S = torch.stack(S, dim=1)
        return S

    def loss(self) -> Callable:
        def func(model, m, S, progress):
            _ = progress
            S_tilde = model(m)
            # Sum of channel wise cross-entropy
            ce_S = multi_cross_entropy(S_tilde, S)

            S_tilde = S_tilde.argmax(dim=2)

            print()
            print()
            print()
            import ipdb; ipdb.set_trace()
            print()
            print()
            print()

            loss = ce_S
            return loss

        return func
