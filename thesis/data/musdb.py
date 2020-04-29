import torch
import musdb

from ..data import Dataset
from ..nn.modules import MelSpectrogram


class MusDB(Dataset):
    def __init__(self, path: str, subsets: str):
        super(MusDB, self).__init__()

        self.db = musdb.DB(root=path, subsets=subsets)
        self.spectrograph = MelSpectrogram()

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx: int):
        track = self.db[idx]
        signals = torch.tensor(track.stems.mean(-1), dtype=torch.float32)
        return signals
