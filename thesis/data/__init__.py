from torch.utils import data
from ..nn.modules import MelSpectrogram


class Dataset(data.Dataset):
    def __init__(self, sr: int = 14_700, n_mels: int = 80):
        self.rate = sr
        self.spectrograph = MelSpectrogram(n_mels=n_mels, sr=sr)

    def loader(self, batch_size: int, shuffle=True, **kwargs) -> data.DataLoader:
        return data.DataLoader(self, batch_size=batch_size, num_workers=8, shuffle=shuffle, **kwargs)

    def __str__(self) -> str:
        return f"{type(self).__name__} with <{len(self):>7} signals>"

    def _mel_get(self, signal, do_time, do_mel):
        if do_mel:
            mel = self.spectrograph(signal.squeeze())
            if do_time:
                return signal, mel
            else:
                return mel
        else:
            return signal
