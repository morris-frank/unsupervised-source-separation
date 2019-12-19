import random
from math import ceil
from typing import Optional, Tuple

import numpy as np
import torch
from torch import dtype as torch_dtype
from torch.utils import data

from nsynth.functional import encode_μ_law


class ToyDataSet(data.Dataset):
    def __init__(self, filepath: str, μ: int = 100, crop: Optional[int] = None,
                 dtype: torch_dtype = torch.float32):
        self.data = np.load(filepath, allow_pickle=True)
        self.μ, self.crop = μ, crop
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return f'ToyDataSet <{len(self):>7} signals>'

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]

        mix = torch.tensor(item['mix'], dtype=self.dtype)
        mix = encode_μ_law(mix, μ=self.μ - 1) / ceil(self.μ / 2)

        sources = torch.tensor(item['sources'], dtype=self.dtype)
        sources = (encode_μ_law(sources, μ=self.μ - 1) + ceil(
            self.μ / 2)).long()

        if self.crop:
            p = random.randint(0, mix.numel() - self.crop)
            mix = mix[p:p + self.crop]
            sources = sources[:, p:p + self.crop]

        return mix.unsqueeze(0), sources

    def loader(self, nbatch) -> data.DataLoader:
        return data.DataLoader(self, batch_size=nbatch, num_workers=8,
                               shuffle=True)
