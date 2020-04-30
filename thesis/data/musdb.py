import torch
import musdb

from ..data import Dataset
from ..functional import normalize
from random import randint


class MusDB(Dataset):
    def __init__(self, path: str, subsets: str, mel: bool = False, **kwargs):
        super(MusDB, self).__init__(sr=14700, **kwargs)

        self.mel = mel
        self.db = musdb.DB(root=path, subsets=subsets)
        self.L = 3 * 2**10

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx: int):
        track = self.db[idx]

        stems = track.stems[1:, ::3, :]
        ν = randint(0, stems.shape[1] - self.L)
        stems = stems[:, None, ν:ν+self.L, :].mean(-1)

        signals = torch.tensor(stems, dtype=torch.float32)
        for i in range(4):
            signals[i, :] = normalize(signals[i, :])
        signals = self._mel_get(signals, self.mel)
        return signals
