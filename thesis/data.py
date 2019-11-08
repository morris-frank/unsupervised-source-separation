import random

import musdb
import numpy as np
import torch.utils.data
from torch import dtype as _dtype
from typing import Tuple

from .utils import stereo_to_mono


class MusDB(torch.utils.data.Dataset):
    def __init__(self,
                 root: str = '/home/morris/var/data/musdb18/',  # TODO: REMOVE LATER ON
                 subset: str = 'train',
                 split: str = 'train',
                 random_mix: bool = True,
                 stereo: bool = False,
                 is_wav: bool = False,
                 sample_length: int = 10,
                 samples_per_track: int = 64,
                 dtype: _dtype = torch.float32):
        self.dtype = dtype
        self.sample_length = sample_length  # in seconds
        self.samples_per_track = samples_per_track
        self.stereo = stereo
        self.random_mix = random_mix
        self.sample_rate = 44100  # MusDB Sample Rate is fixed
        self.split = split

        self.db = musdb.DB(root=root, is_wav=is_wav, subsets=subset, split=split)

    def __len__(self) -> int:
        return len(self.db.tracks) * self.samples_per_track

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Item enumerates the samples ⇒
        track = self.db.tracks[item // self.samples_per_track]

        # During training we have samples, during testing full songs:
        sample_length = self.sample_length if self.split == 'train' else track.duration

        # In testing split sample start will always be zero
        track.chunk_start = random.uniform(0, track.duration - sample_length)

        source_tracks = []
        for source in self.db.setup['sources']:
            if self.split == 'train' and self.random_mix:
                # TODO: Problem with random mix – some songs (music delta) are very short < 60 sec
                # If doing random mix, we choose a random song for each source track
                track = random.choice(self.db.tracks)
                track.chunk_start = random.uniform(0, track.duration - sample_length)
            track.chunk_duration = sample_length  # again = duration in testing split

            source_tracks.append(self.preprocess_audio(track.sources[source].audio))
        y = torch.stack(source_tracks, dim=0)

        # During training we assemble our own mix, in testing pick the mixed stem:
        if self.split == 'train':
            x = torch.sum(y, dim=0)
        else:
            x = self.preprocess_audio(track.audio)

        return x, y

    def preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        """
        Preprocess an Numpy audio stream from MusDB to a Tensor
        :param audio:
        :return: the audio as tensor
        """
        if audio.shape[0] != 2:
            audio = audio.T
        tensor = torch.tensor(audio, dtype=self.dtype)
        if not self.stereo:
            tensor = stereo_to_mono(tensor)
        return tensor
