from typing import Any

import torch
from torch import nn

from ...utils import _LossLogger


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.ℒ = _LossLogger()

    def loss(self, x: Any, y: Any) -> torch.Tensor:
        raise NotImplementedError
