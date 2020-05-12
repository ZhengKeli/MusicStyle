import os
import unittest
import numpy as np
import matplotlib.pyplot as plt

from dataset import load_audio, play_audio, save_audio


class TestDatasetAudio(unittest.TestCase):
    def setUp(self):
        self.load_audio_path = r"C:\Users\keli\OneDrive\学习\CIS Machine Learning\MusicStyle\data\genres\blues\blues.00000.wav"
        self.save_audio_path = "./beat.mp3"
        
        self.sample_rate = 22050
        ts = np.arange(0, 5 * self.sample_rate) / self.sample_rate
        self.play_audio = (np.sin(ts * 2 * np.pi * 599) + np.sin(ts * 2 * np.pi * 601)) / 3
    
    def test_load_audio(self):
        audio = load_audio(self.load_audio_path, self.sample_rate)
        print("loaded audio from file " + self.save_audio_path)
        plt.plot(audio)
        plt.show()
    
    def test_save_audio(self):
        save_audio(self.play_audio, self.save_audio_path, self.sample_rate)
        self.assertTrue(os.path.exists(self.save_audio_path))
        print("saved audio to file " + self.save_audio_path)
        os.remove(self.save_audio_path)
        print("deleted audio file " + self.save_audio_path)
    
    def test_play_audio(self):
        play_audio(self.play_audio, self.sample_rate)
        print("played audio")
