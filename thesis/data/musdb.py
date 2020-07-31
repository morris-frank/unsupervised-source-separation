from glob import glob
from random import randint

import librosa
import musdb
import numpy as np
import torch
from torch.nn import functional as F

from ..data import Dataset
from ..functional import normalize


class MusDB(Dataset):
    def __init__(self, path: str, subsets: str, mel: bool = False):
        super(MusDB, self).__init__(sr=14_700, n_mels=128)
        self.path, self.subsets = path, subsets
        self.mel = mel
        self.db = musdb.DB(root=path, subsets=subsets)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx: int):
        track = self.db[idx]
        # Take only the left channel, do not take mean, cause of weirdness
        stems = track.stems[1:, :, 0]
        stems = np.asfortranarray(stems)
        # Down sample to our sample rate
        stems = librosa.resample(stems, track.rate, self.rate, res_type="polyphase")
        signals = torch.tensor(stems, dtype=torch.float32)
        for i in range(4):
            signals[i, :] = normalize(signals[i, :])
        signals = self._mel_get(signals, True, self.mel)
        return signals

    def pre_save(self, n_per_song: int, length: float):
        for i, (wav, mel) in enumerate(self):
            c = mel.shape[2] / wav.shape[1]
            for _ in range(n_per_song):
                ν = randint(0, wav.shape[1] - length)
                _mel = mel[:, :, int(ν * c) : int((ν + length) * c)]
                _wav = wav[:, ν : ν + length]
                yield _wav, _mel


class MusDBSamples(Dataset):
    def __init__(self, path: str, subsets: str, form: str, length: int = False, **kwargs):
        super(MusDBSamples, self).__init__(**kwargs)
        assert form in ("mel", "wav")
        path = path + "_samples/" + subsets + f"/*{form}.pt"
        self.files = glob(path)
        self.length = length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        s, mel = torch.load(self.files[idx])
        s = s.squeeze()
        return s.contiguous(), mel.contiguous()
