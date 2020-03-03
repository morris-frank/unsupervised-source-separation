from abc import ABC

import torch
from torch import nn

from ...utils import _LossLogger


class BaseModel(ABC, nn.Module):
    def __init__(self, name: str = ""):
        super(BaseModel, self).__init__()
        self.ℒ = _LossLogger()
        self.name = name

    def test(self, *args) -> torch.Tensor:
        pass

    def infer(self, *args, **kwargs) -> torch.Tensor:
        pass
