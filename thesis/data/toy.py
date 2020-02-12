import random
from glob import glob
from typing import Tuple, Optional

import numpy as np
import torch
from librosa import stft
from torch.utils import data

from ..data import Dataset
from ..functional import encode_μ_law


def _prepare_toy_audio_data(
    mix: np.ndarray,
    sources: np.ndarray,
    crop: int,
    offset: int = 0,
    μ: Optional[int] = None,
):
    mix = torch.tensor(mix, dtype=torch.float32)
    if μ:
        assert μ & 1
        hμ = (μ - 1) // 2
        mix = encode_μ_law(mix, μ=μ) / hμ

    sources = torch.tensor(sources, dtype=torch.float32)
    if μ:
        sources = encode_μ_law(sources, μ=μ)
        sources = (sources + hμ).long()

    mix = mix[offset : offset + crop]
    sources = sources[:, offset : offset + crop]
    return mix.unsqueeze(0), sources


class ToyData(Dataset):
    """
    A Dataset that loads that loads the toy data. Mix + all sources
    cropped.
    """

    def __init__(self, filepath: str, crop: int, μ: Optional[int] = None):
        self.data = np.load(filepath, allow_pickle=True)
        self.μ, self.crop = μ, crop

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        p = random.randint(0, item["mix"].size - self.crop)
        mix, sources = _prepare_toy_audio_data(
            item["mix"], item["sources"], self.crop, p, self.μ
        )
        return mix, sources


class ToyDataSpectral(ToyData):
    def __init__(self, *args, **kwargs):
        super(ToyDataSpectral, self).__init__(*args, **kwargs)
        self.n_fft = 2 ** 7
        self.f = lambda x: np.abs(stft(x, n_fft=self.n_fft))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        p = random.randint(0, item["mix"].size - self.crop)
        mix, _sources = item["mix"], item["sources"]
        mix = mix[p:p+self.crop]
        _sources = _sources[:, p:p+self.crop]
        mix = self.f(mix)
        mix = torch.tensor(mix, dtype=torch.float32).unsqueeze(0)
        sources = []
        for i in range(_sources.shape[0]):
            sources.append(torch.tensor(self.f(_sources[i, :]), dtype=torch.float32))
        return mix, torch.stack(sources)


class ToyDataSingle(ToyData):
    def __init__(self, *args, **kwargs):
        super(ToyDataSingle, self).__init__(*args, **kwargs)

    def __getitem__(self, item: int) -> Tuple[Tuple[torch.Tensor, int], torch.Tensor]:
        mix, sources = super(ToyDataSingle, self).__getitem__(item)
        t = random.randint(0, sources.shape[0] - 1)
        source = sources[None, t, :]
        return (mix, t), source


class ToyDataSequential(Dataset):
    def __init__(
        self,
        filepath: str,
        crop: int,
        batch_size: int,
        steps: int = 5,
        stride: int = None,
        μ: Optional[int] = None,
    ):
        self.μ, self.crop, self.steps = μ, crop, steps
        self.ifile, self.data = None, None
        self.batch_size = batch_size
        self.files = sorted(glob(filepath))
        self.load_file(0)
        if not stride:
            self.stride = crop // 2

    def load_file(self, i):
        self.ifile = i
        self.data = np.load(self.files[i], allow_pickle=True)

    def __len__(self) -> int:
        # the minus 1 stays unexplained
        return len(self.files) * len(self.data) * self.steps - (
            self.steps * self.batch_size
        )

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, int], torch.Tensor]:
        i_batch = idx // (self.batch_size * self.steps)
        i_in_batch = idx % self.batch_size
        i_sample = i_batch * self.batch_size + i_in_batch

        # Index of file where sample is
        i_file = i_sample // len(self.data)
        # Index of sample inside this file
        i_sample_in_file = i_sample % len(self.data)

        # If we are currently in the wrong file, load the next one
        if i_file != self.ifile:
            self.load_file(i_file)
        item = self.data[i_sample_in_file]

        offset = idx // self.batch_size % self.steps
        offset *= self.stride

        mix, sources = _prepare_toy_audio_data(
            item["mix"], item["sources"], self.crop, offset, self.μ
        )
        return (mix, offset), sources

    def loader(self, batch_size: int) -> data.DataLoader:
        return data.DataLoader(
            self, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=True
        )
