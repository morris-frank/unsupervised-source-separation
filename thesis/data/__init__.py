from torch.utils import data
from torch.nn import functional as F
from ..nn.modules import MelSpectrogram
from torchaudio.transforms import Spectrogram


class Dataset(data.Dataset):
    def __init__(self, interpolate: bool = False, sr: int = 16000, complex: bool = False, n_mels: int = 80):
        self.interpolate, self.complex = interpolate, complex

        if complex:
            self.spectrograph = Spectrogram(n_fft=128, power=None, normalized=True)
            # inverse is:
            # waveform = istft(spectrogram, n_fft=128, length=3072, normalized=True)
        else:
            self.spectrograph = MelSpectrogram(n_mels=n_mels, sr=sr)

    def loader(self, batch_size: int, shuffle=True, **kwargs) -> data.DataLoader:
        return data.DataLoader(self, batch_size=batch_size, num_workers=8, shuffle=shuffle, **kwargs)

    def __str__(self) -> str:
        return f"{type(self).__name__} with <{len(self):>7} signals>"

    def _mel_get(self, signal, do_time, do_mel):
        if do_mel:
            mel = self.spectrograph(signal.squeeze())
            if self.complex:
                N, _, L, _ = mel.shape
                mel = mel.permute(0, 1, 3, 2).view(N, -1, L)
            if self.interpolate:
                if mel.ndim == 2:
                    mel = mel[None, ...]
                mel = F.interpolate(mel, signal.shape[-1], mode="linear", align_corners=False)
            mel = F.interpolate(mel, 2**9, mode="linear", align_corners=False)
            if do_time:
                return signal, mel
            else:
                return mel
        else:
            return signal
