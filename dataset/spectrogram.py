import librosa
import numpy as np


def cqt_spectrogram(wave: np.ndarray, sample_rate=22050, n_cqt=84, strides=512,
                    norm=True, **kwargs):
    sp = librosa.cqt(wave, sample_rate, strides, n_bins=n_cqt, **kwargs)
    sp = np.abs(sp)
    sp = librosa.amplitude_to_db(sp, ref=np.max)
    
    if norm:
        sp -= np.mean(sp)
        sp /= np.sqrt(np.mean(np.square(sp)))
    
    return sp


def mfcc_spectrogram(wave: np.ndarray, sample_rate=22050, n_mfcc=64,
                     norm_chan=True, norm_all=False, flip=True, **kwargs):
    sp = librosa.feature.mfcc(wave, sample_rate, n_mfcc=n_mfcc, **kwargs)
    
    if norm_chan:
        sp -= np.mean(sp, -1, keepdims=True)
        sp /= np.sqrt(np.mean(np.square(sp), -1, keepdims=True))
    
    if norm_all:
        sp -= np.mean(sp)
        sp /= np.sqrt(np.mean(np.square(sp)))
    
    if flip:
        sp = np.flip(sp, 0)
    
    return sp
