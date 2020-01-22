import random
from glob import glob
from math import ceil
from typing import Tuple

import numpy as np
import torch
from torch.utils import data

from nsynth.functional import encode_μ_law


def _prepare_toy(mix: np.ndarray, sources: np.ndarray, μ: int, crop: int,
                 offset: int = 0):
    mix = torch.tensor(mix, dtype=torch.float32)
    mix = encode_μ_law(mix, μ=μ - 1) / ceil(μ / 2)

    sources = torch.tensor(sources, dtype=torch.float32)
    sources = (encode_μ_law(sources, μ=μ - 1) + ceil(
        μ / 2)).long()

    mix = mix[offset:offset + crop]
    sources = sources[:, offset:offset + crop]
    return mix.unsqueeze(0), sources


class ToyData(data.Dataset):
    def __init__(self, filepath: str, μ: int, crop: int):
        self.data = np.load(filepath, allow_pickle=True)
        self.μ, self.crop = μ, crop

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return f'ToyData <{len(self):>7} signals>'

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]

        p = random.randint(0, item['mix'].numel() - self.crop)
        mix, sources = _prepare_toy(item['mix'], item['sources'], self.μ,
                                    self.crop, p)
        return mix, sources

    def loader(self, nbatch: int) -> data.DataLoader:
        return data.DataLoader(self, batch_size=nbatch, num_workers=8,
                               shuffle=True)


class ToyDataSequential(data.Dataset):
    def __init__(self, filepath: str, μ: int, crop: int,
                 nbatch: int, steps: int = 5, stride: int = None):
        self.μ, self.crop, self.steps, self.nbatch = μ, crop, steps, nbatch
        self.files = sorted(glob(filepath))
        self.load_file(0)
        if not stride:
            self.stride = crop // 2

    def load_file(self, i):
        self.ifile = i
        self.data = np.load(self.files[i], allow_pickle=True)

    def __len__(self) -> int:
        # the minus 1 stays unexplained
        return len(self.files) * len(self.data) * self.steps \
               - (self.steps * self.nbatch)

    def __str__(self) -> str:
        return f'ToyDataSequential <{len(self):>7} signals>'

    def loader(self, nbatch: int) -> data.DataLoader:
        return data.DataLoader(self, batch_size=nbatch, num_workers=0,
                               shuffle=False, drop_last=True)

    def __getitem__(self, idx: int) \
            -> Tuple[Tuple[torch.Tensor, int], torch.Tensor]:
        i_batch = idx // (self.nbatch * self.steps)
        i_in_batch = idx % self.nbatch
        i_sample = i_batch * self.nbatch + i_in_batch

        # Index of file where sample is
        i_file = i_sample // len(self.data)
        # Index of sample inside this file
        i_sample_in_file = i_sample % len(self.data)

        # If we are currently in the wrong file, load the next one
        if i_file != self.ifile:
            self.load_file(i_file)
        item = self.data[i_sample_in_file]

        offset = idx // self.nbatch % self.steps
        offset *= self.stride

        mix, sources = _prepare_toy(item['mix'], item['sources'], self.μ,
                                    self.crop, offset)
        return (mix, offset), sources


class ToyDataSingle(ToyData):
    def __init__(self, *args, **kwargs):
        super(ToyDataSingle, self).__init__(*args, **kwargs)

    def __getitem__(self, item: int) \
            -> Tuple[Tuple[torch.Tensor, int], torch.Tensor]:
        mix, sources = super(ToyDataSingle, self).__getitem__(item)
        t = random.randint(0, sources.shape[0] - 1)
        source = sources[None, t, :]
        return (mix, t), source
