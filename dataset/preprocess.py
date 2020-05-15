import librosa
import numpy as np


def constant_quality_transform(wave: np.ndarray, sample_rate=22050, strides=512, n_bins=84, **kwargs):
    spectrogram = librosa.cqt(wave, sample_rate, strides, n_bins=n_bins, **kwargs)
    spectrogram = np.abs(spectrogram)
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    return spectrogram