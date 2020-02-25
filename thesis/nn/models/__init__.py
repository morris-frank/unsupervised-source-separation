from abc import ABC
from typing import Any

import torch
from torch import nn

from ...utils import _LossLogger


class BaseModel(ABC, nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.ℒ = _LossLogger()
        self.name = ""

    def test(self, x: Any, y: Any) -> torch.Tensor:
        pass

    def infer(self, m: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass
