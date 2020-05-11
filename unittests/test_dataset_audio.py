import unittest
import numpy as np
import matplotlib.pyplot as plt

from dataset import load_audio, play_audio, save_audio


class TestDatasetAudio(unittest.TestCase):
    def setUp(self):
        self.audio_path = r"C:\Users\keli\OneDrive\学习\CIS Machine Learning\MusicStyle\data\genres\blues\blues.00000.wav"
        
        sample_rate = 22050
        ts = np.arange(0, 5 * sample_rate) / sample_rate
        self.ys = (np.sin(ts * 2 * np.pi * 599) + np.sin(ts * 2 * np.pi * 601)) / 3
        self.sample_rate = sample_rate
    
    def test_load_audio(self):
        audio = load_audio(self.audio_path)
        plt.plot(audio)
        plt.show()
    
    def test_save_audio(self):
        save_audio(self.ys, "./beat.mp3", self.sample_rate)
    
    def test_play_audio(self):
        play_audio(self.ys, self.sample_rate)
