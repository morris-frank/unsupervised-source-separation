import torch
import musdb

from ..data import Dataset
from ..functional import normalize


class MusDB(Dataset):
    def __init__(self, path: str, subsets: str, mel: bool = False, **kwargs):
        super(MusDB, self).__init__(sr=44100, **kwargs)

        self.mel = mel
        self.db = musdb.DB(root=path, subsets=subsets)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx: int):
        track = self.db[idx]
        signals = torch.tensor(track.stems.mean(-1), dtype=torch.float32)
        for i in range(5):
            signals[i, :] = normalize(signals[i, :])
        signals = self._mel_get(signals, self.mel)
        return signals
