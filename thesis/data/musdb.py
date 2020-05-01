import os
from os import path
from random import randint
from itertools import product

import musdb
import torch
from tqdm import tqdm

from ..data import Dataset
from ..functional import normalize


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
        for j, i in tqdm(product(range(n), range(len(self)))):
            signals = self[i]
            torch.save(signals, f"{fp}/{i:03}_{j:03}_{os.getpid()}.pt")
