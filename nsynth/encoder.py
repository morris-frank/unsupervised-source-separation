import torch
from torch import nn
from torch.nn import functional as F

from .functional import dilate, range_product


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 16,
                 n_blocks: int = 3, n_layers: int = 10, width: int = 128,
                 kernel_size: int = 3, pool_size: int = 512):
        """

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            n_blocks: Number of dilation blocks / stages
            n_layers: Number of layers in each stage / block
            width: Width of the hidden layers
            kernel_size: General Kernel size for the convs
            pool_size: Size of avg pooling before the output
        """
        super(TemporalEncoder, self).__init__()

        assert kernel_size % 2 != 0
        pad = (kernel_size - 1) // 2

        # Go from input channels to hidden width:
        self.init = nn.Conv1d(in_channels, width, kernel_size, padding=pad)
        # Go from hidden width to final output channels
        self.final = nn.Sequential(
            nn.Conv1d(width, out_channels, 1),
            nn.AvgPool1d(kernel_size=pool_size)
        )

        self.residuals = []
        for _, layer in range_product(n_blocks, n_layers):
            self.residuals.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(width, width, kernel_size, padding=pad),
                nn.ReLU(),
                nn.Conv1d(width, width, 1)
            ))
        self.residuals = nn.ModuleList(self.residuals)

        # TODO: replace with prime factors
        self.dilations = [2 ** l for _, l in range_product(n_blocks, n_layers)]
        self.dilations.append(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.init(x)
        for i, residual in enumerate(self.residuals):
            # Increase dilation by one step
            y = dilate(y, new=self.dilations[i], old=self.dilations[i - 1])
            y = y + residual(y)
        # Remove the compound dilations
        y = dilate(y, new=self.dilations[-1], old=self.dilations[-2])
        y = self.final(y)
        return y


class ConditionalTemporalEncoder(nn.Module):
    def __init__(self, n_classes: int, in_channels: int = 1,
                 out_channels: int = 16, n_blocks: int = 3, n_layers: int = 10,
                 width: int = 128, kernel_size: int = 3, pool_size: int = 512,
                 device: str = 'cpu'):
        super(ConditionalTemporalEncoder, self).__init__()

        assert kernel_size % 2 != 0
        pad = (kernel_size - 1) // 2

        self.n_classes = n_classes
        self.device = device

        self.init = nn.Conv1d(in_channels, width, kernel_size, padding=pad)
        self.final = nn.Sequential(
            nn.Conv1d(width, out_channels, 1),
            nn.AvgPool1d(kernel_size=pool_size)
        )

        self.residuals_front, self.residuals_back = [], []
        self.conditionals = []
        for _, layer in range_product(n_blocks, n_layers):
            self.residuals_front.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(width, width, kernel_size, padding=pad)
            ))
            self.conditionals.append(nn.Sequential(
                nn.Linear(n_classes, width, bias=False)
            ))
            self.residuals_back.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(width, width, 1)
            ))
        self.residuals_front = nn.ModuleList(self.residuals_front)
        self.residuals_back = nn.ModuleList(self.residuals_back)
        self.conditionals = nn.ModuleList(self.conditionals)

        # TODO: replace with prime factors
        self.dilations = [2 ** l for _, l in
                          range_product(n_blocks, n_layers)]
        self.dilations.append(1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == labels.numel()
        targets = F.one_hot(labels, self.n_classes).float().to(self.device)

        y = self.init(x)
        for i, (front, cond, back) in enumerate(
                zip(self.residuals_front, self.conditionals,
                    self.residuals_back)):
            # Increase dilation by one step
            y = dilate(y, new=self.dilations[i], old=self.dilations[i - 1])
            c = cond(targets)[..., None]
            # As we compound dilate the y we have to repeat the conditional
            # along the batch dimension
            c = c.repeat(y.shape[0] // c.shape[0], 1, 1)
            y = y + back(front(y) + c)
        # Remove the compound dilations
        y = dilate(y, new=self.dilations[-1], old=self.dilations[-2])
        y = self.final(y)
        return y
