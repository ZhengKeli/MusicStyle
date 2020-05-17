import unittest

import matplotlib.pyplot as plt
import numpy as np

from dataset.audio import load_audio
from dataset.dataset import scan_dataset
from dataset.spectrogram import cqt_spectrogram, mfcc_spectrogram


class TestDatasetSpectrogram(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = r"../data"
        self.sample_rate = 22050
    
    def wave_gen(self, num_each_cls=3):
        dataset = scan_dataset(self.dataset_dir)
        for cls, fn_list in dataset.items():
            for fi, fn in enumerate(fn_list):
                if fi >= num_each_cls:
                    break
                wave = load_audio(fn, self.sample_rate)
                yield wave, cls, fi
    
    def show_spectrogram(self, sp, cls, fi):
        plt.figure(figsize=(14, 6))
        plt.imshow(sp)
        plt.xlim(0, 400)
        plt.title('spectrogram: ' + cls + ' ' + str(fi))
        plt.tight_layout()
        plt.show()
    
    def show_hsit(self, sp, cls, fi):
        plt.figure()
        plt.hist(np.reshape(sp, [-1]), 20)
        plt.title('histogram: ' + cls + ' ' + str(fi))
        plt.tight_layout()
        plt.show()
    
    def test_cqt_spectrogram(self):
        for wave, cls, fi in self.wave_gen(5):
            sp = cqt_spectrogram(wave, self.sample_rate, n_cqt=84)
            # self.show_hsit(sp, cls, fi)
            self.show_spectrogram(sp, cls, fi)
    
    def test_mfcc_spectrogram(self):
        for wave, cls, fi in self.wave_gen(3):
            sp = mfcc_spectrogram(wave, self.sample_rate, n_mfcc=84)
            # self.show_hsit(sp, cls, fi)
            self.show_spectrogram(sp, cls, fi)
