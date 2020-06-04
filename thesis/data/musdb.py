import os
from glob import glob
from itertools import product
from os import path
from random import randint

import musdb
import torch

from ..data import Dataset
from ..functional import normalize
from torch.nn import functional as F


class MusDB(Dataset):
    def __init__(self, path: str, subsets: str, mel: bool = False, **kwargs):
        super(MusDB, self).__init__(sr=14700, **kwargs)
        self.path, self.subsets = path, subsets
        self.mel = mel
        self.db = musdb.DB(root=path, subsets=subsets)
        self.L = 3 * 2 ** 10

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx: int):
        track = self.db[idx]

        stems = track.stems[1:, ::3, :]
        ν = randint(0, stems.shape[1] - self.L)
        stems = stems[:, None, ν : ν + self.L, :].mean(-1)

        signals = torch.tensor(stems, dtype=torch.float32)
        for i in range(4):
            signals[i, :] = normalize(signals[i, :])
        signals = self._mel_get(signals, self.mel)
        return signals

    def pre_save(self, n: int):
        fp = path.normpath(self.path) + "_samples/" + self.subsets
        os.makedirs(fp, exist_ok=True)
        for j, i in product(range(n), range(len(self))):
            if i == 0:
                print(f"Start next round for: {os.getpid()}")
            signals = self[i]
            torch.save(signals, f"{fp}/{i:03}_{j:03}_{os.getpid()}.pt")


class MusDBSamples(Dataset):
    def __init__(self, path: str, subsets: str, **kwargs):
        super(MusDBSamples, self).__init__(**kwargs)
        path = path + "_samples/" + subsets + "/*pt"
        self.files = glob(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        s, mel = torch.load(self.files[idx])
        s = s.squeeze()
        mel = F.interpolate(mel, s.shape[-1], mode="linear",
                            align_corners=False)
        return s.contiguous(), mel.contiguous()


class MusDBSamples2(MusDBSamples):
    def __len__(self):
        return 4 * super(MusDBSamples2, self).__len__()

    def __getitem__(self, idx: int):
        _, mel = torch.load(self.files[idx//4])
        i = randint(0, 3)
        mel = mel[i, ...]
        i = torch.empty(1, mel.shape[-1]).fill_(i)
        return mel, i



