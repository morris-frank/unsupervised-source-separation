from glob import glob
from random import randint

import librosa
import musdb
import numpy as np
import torch

from ..data import Dataset
from ..functional import normalize


class MusDB(Dataset):
    def __init__(self, path: str, subsets: str, mel: bool = False):
        super(MusDB, self).__init__(sr=24_000, n_mels=265)
        self.path, self.subsets = path, subsets
        self.mel = mel
        self.db = musdb.DB(root=path, subsets=subsets)
        self.time_sr = 14_700

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx: int):
        track = self.db[idx]
        # Take only the left channel, do not take mean, cause of weirdness
        stems = track.stems[1:, :, 0]
        stems = np.asfortranarray(stems)

        # Down sample to our sample rate
        mel_stems = librosa.resample(stems, track.rate, self.rate, res_type="polyphase")
        time_stems = librosa.resample(stems, track.rate, self.time_sr, res_type="polyphase")

        mel = self.spectrograph(torch.tensor(mel_stems, dtype=torch.float32))

        wav = torch.tensor(time_stems, dtype=torch.float32)
        for i in range(4):
            wav[i, :] = normalize(wav[i, :])
        return wav, mel

    def pre_save(self, n_per_song: int, length: float):
        for i, (wav, mel) in enumerate(self):
            c = mel.shape[2] / wav.shape[1]
            for _ in range(n_per_song):
                ν = randint(0, wav.shape[1] - length)
                _mel = mel[:, :, int(ν * c) : int((ν + length) * c)]
                _wav = wav[:, ν : ν + length]
                yield _wav, _mel


class MusDBSamples(Dataset):
    def __init__(
        self, path: str, subsets: str, space: str, length: int = False
    ):
        super(MusDBSamples, self).__init__()
        assert space in ("mel", "time")
        path = path + "_samples/" + subsets + f"/*_{space}.npy"
        self.files = glob(path)
        self.length = length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        x = np.load(self.files[idx])
        x = torch.from_numpy(x)
        return x
