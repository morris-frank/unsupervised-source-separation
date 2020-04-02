import random
from glob import glob
from typing import Optional

import numpy as np
import torch

from ..data import Dataset
from typing import Union


class ToyData(Dataset):
    def __init__(
        self,
        path: str,
        mix: bool = False,
        mel: bool = False,
        source: Union[bool, int] = False,
        mel_source: bool = False,
        rand_amplitude: Optional[float] = None,
        crop: Optional[int] = None,
    ):
        super(ToyData, self).__init__()
        self.files = glob(f"{path}/*npy")
        self.rand_amplitude, self.crop = rand_amplitude, crop

        self.mix, self.mel = mix, mel

        self.k = "all" if isinstance(source, bool) else source
        self.source = source is not False
        self.mel_source = mel_source

        assert mix or self.source
        assert mix or not mel
        assert self.source or not mel_source
        assert self.source or not rand_amplitude

    def __len__(self):
        return len(self.files)

    def get(self, idx: int):
        datum = np.load(self.files[idx], allow_pickle=True).item()

        if self.crop:
            mix_w = datum["mix"].size
            p = random.randint(0, mix_w - self.crop)
            datum["mix"] = datum["mix"][p : p + self.crop]
            datum["sources"] = datum["sources"][:, p : p + self.crop]
            mel_w = datum["mel_mix"].shape[0]
            l_m, r_m = int(p / mix_w * mel_w), int((p + self.crop) / mix_w * mel_w)
            datum["mel_mix"] = datum["mel_mix"][l_m : r_m + 1, :]
            datum["mel_sources"] = datum["mel_sources"][:, l_m : r_m + 1, :]

        datum["mix"] = torch.tensor(datum["mix"], dtype=torch.float32).unsqueeze(0)
        datum["sources"] = torch.tensor(datum["sources"], dtype=torch.float32)
        datum["mel_mix"] = torch.tensor(
            datum["mel_mix"], dtype=torch.float32
        ).transpose(0, 1)
        datum["mel_sources"] = torch.tensor(
            datum["mel_sources"], dtype=torch.float32
        ).transpose(1, 2)

        return datum

    @staticmethod
    def _mel_get(datum, name, mel):
        if mel:
            return datum[name].contiguous(), datum[f"mel_{name}"].contiguous()
        else:
            return datum[name].contiguous()

    def __getitem__(self, idx: int):
        datum = self.get(idx)

        if self.k != "all":
            datum["sources"] = datum["sources"][None, self.k, :].contiguous()
            datum["mel_sources"] = datum["mel_sources"][self.k, ...].contiguous()

        if self.rand_amplitude:
            A = torch.rand(datum["sources"].shape[0], 1) * self.rand_amplitude + (
                1.0 - self.rand_amplitude
            )
            datum["sources"] = A * datum["sources"]
            datum["mix"] = datum["sources"].mean(1, keepdim=True)

        if self.source:
            sources = self._mel_get(datum, "sources", self.mel_source)

        if self.mix:
            mix = self._mel_get(datum, "mix", self.mel)
            if self.source:
                return mix, sources
            return mix
        else:
            return sources
