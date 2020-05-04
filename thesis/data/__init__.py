from torch.utils import data
from torch.nn import functional as F
from ..nn.modules import MelSpectrogram


class Dataset(data.Dataset):
    def __init__(self, interpolate: bool = False, sr: int = 16000):
        self.interpolate = interpolate
        self.spectrograph = MelSpectrogram(sr=sr)

    def loader(self, batch_size: int, **kwargs) -> data.DataLoader:
        return data.DataLoader(self, batch_size=batch_size, num_workers=8, shuffle=True, **kwargs)

    def __str__(self) -> str:
        return f"{type(self).__name__} with <{len(self):>7} signals>"

    def _mel_get(self, signal, compute_mel):
        if compute_mel:
            mel = self.spectrograph(signal.squeeze())
            if self.interpolate:
                mel = F.interpolate(mel, signal.shape[-1], mode="linear", align_corners=False)
            return signal, mel
        else:
            return signal
