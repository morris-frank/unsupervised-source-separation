from torch import nn
import torch

from ..modules import Flatten


class ComponentVAE(nn.Module):

    def __init__(self, input_nc, z_dim=16, full_res=False):
        super().__init__()
        self._input_nc = input_nc
        self._z_dim = z_dim
        # full_res = False # full res: 128x128, low res: 64x64
        h_dim = 4096 if full_res else 1024
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc + 1, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(h_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 32)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim + 2, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, input_nc + 1, 1),
        )
        self._bg_logvar = 2 * torch.tensor(0.09).log()
        self._fg_logvar = 2 * torch.tensor(0.11).log()

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    @staticmethod
    def spatial_broadcast(z, h, w):
        # Batch size
        n = z.shape[0]
        # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
        z_b = z.view((n, -1, 1, 1)).expand(-1, -1, h, w)
        # Coordinate axes:
        x = torch.linspace(-1, 1, w, device=z.device)
        y = torch.linspace(-1, 1, h, device=z.device)
        y_b, x_b = torch.meshgrid(y, x)
        # Expand from (h, w) -> (n, 1, h, w)
        x_b = x_b.expand(n, 1, -1, -1)
        y_b = y_b.expand(n, 1, -1, -1)
        # Concatenate along the channel dimension: final shape = (n, z_dim + 2, h, w)
        z_sb = torch.cat((z_b, x_b, y_b), dim=1)
        return z_sb

    def forward(self, x, log_m_k, background=False):
        """
        :param x: Input image
        :param log_m_k: Attention mask logits
        :return: x_k and reconstructed mask logits
        """
        params = self.encoder(torch.cat((x, log_m_k), dim=1))
        z_mu = params[:, :self._z_dim]
        z_logvar = params[:, self._z_dim:]
        z = self.reparameterize(z_mu, z_logvar)

        # "The height and width of the input to this CNN were both 8 larger than the target output (i.e. image) size
        #  to arrive at the target size (i.e. accommodating for the lack of padding)."
        h, w = x.shape[-2:]
        z_sb = self.spatial_broadcast(z, h + 8, w + 8)

        output = self.decoder(z_sb)
        x_mu = output[:, :self._input_nc]
        x_logvar = self._bg_logvar if background else self._fg_logvar
        m_logits = output[:, self._input_nc:]

        return m_logits, x_mu, x_logvar, z_mu, z_logvar
