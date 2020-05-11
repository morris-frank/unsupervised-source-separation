from torch.utils import data
from torch.nn import functional as F
from ..nn.modules import MelSpectrogram
from torchaudio.transforms import Spectrogram


class Dataset(data.Dataset):
    def __init__(self, interpolate: bool = False, sr: int = 16000, complex: bool = False):
        self.interpolate, self.complex = interpolate, complex

        if complex:
            self.spectrograph = Spectrogram(n_fft=1024, hop_length=256, power=None)
        else:
            self.spectrograph = MelSpectrogram(sr=sr)

    def loader(self, batch_size: int, **kwargs) -> data.DataLoader:
        return data.DataLoader(self, batch_size=batch_size, num_workers=8, shuffle=True, **kwargs)

    def __str__(self) -> str:
        return f"{type(self).__name__} with <{len(self):>7} signals>"

    def _mel_get(self, signal, compute_mel):
        if compute_mel:
            mel = self.spectrograph(signal.squeeze())
            if self.complex:
                N, _, L, _ = mel.shape
                mel = mel.permute(0, 1, 3, 2).view(N, -1, L)
            if self.interpolate:
                mel = F.interpolate(mel, signal.shape[-1], mode="linear", align_corners=False)
            return signal, mel
        else:
            return signal
