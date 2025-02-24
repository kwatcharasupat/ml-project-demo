from typing import Tuple
from torch import Tensor
import torchaudio as ta
from torchaudio import functional as taF
from torch.nn import functional as F


class GTZANDataset(ta.datasets.GTZAN):
    __SPLIT_TO_SUBSET__ = {
        "train": "training",
        "val": "validation",
        "test": "testing",
    }

    def __init__(self, data_root: str, split: str):
        super().__init__(
            root=data_root,
            download=False,
            subset=self.__SPLIT_TO_SUBSET__[split],
        )

        self.duration_seconds = 30
        self.target_fs = 22050
        self.duration_samples = self.duration_seconds * self.target_fs

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        try:
            audio, fs, label = super().__getitem__(n)
        except RuntimeError:
            return self.__getitem__(n + 1)

        if fs != self.target_fs:
            audio = taF.resample(audio, orig_freq=fs, new_freq=self.target_fs)

        _, n_samples = audio.shape

        if n_samples < self.duration_samples:
            audio = F.pad(audio, (0, self.duration_samples - n_samples))

        if n_samples > self.duration_samples:
            audio = audio[:, : self.duration_samples]

        return {
            "audio": audio,
            "label": label,
        }


if __name__ == "__main__":
    gtzan = ta.datasets.GTZAN("~/home/kwatchar3/Documents/data/gtzan", download=True)

    for i in range(10):
        waveform, sample_rate, label, track_id, subset = gtzan[i]
        print(waveform.shape, sample_rate, label, track_id, subset)
        print(waveform.min(), waveform.max())
