import unittest

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataset.audio import load_audio
from dataset.dataset import compile_dataset, flatten_dataset, scan_dataset, split_dataset
from dataset.preprocess import constant_quality_transform


class TestDatasetDataset(unittest.TestCase):
    def setUp(self):
        self.dataset_dir = r"C:\Users\keli\OneDrive\学习\CIS Machine Learning\MusicStyle\data\genres"
    
    def test_scan_dataset(self):
        dataset = scan_dataset(self.dataset_dir)
        self.assertTrue(len(dataset.keys()) == 10)
        for type_name, audio_list in dataset.items():
            print(type_name + ":")
            self.assertTrue(len(audio_list) == 100)
            for audio in audio_list:
                print("\t" + audio)
    
    def test_split_dataset(self):
        dataset = scan_dataset(self.dataset_dir)
        subsets = split_dataset(dataset, (3, 1, 1))
        
        print("subset0:")
        for type_name, file_list in subsets[0].items():
            print("\t" + type_name + ":")
            self.assertTrue(len(file_list) == 60)
            for file_path in file_list:
                print("\t\t" + file_path)
        
        print("subset1:")
        for type_name, file_list in subsets[1].items():
            print("\t" + type_name + ":")
            self.assertTrue(len(file_list) == 20)
            for file_path in file_list:
                print("\t\t" + file_path)
        
        print("subset2:")
        for type_name, file_list in subsets[2].items():
            print("\t" + type_name + ":")
            self.assertTrue(len(file_list) == 20)
            for file_path in file_list:
                print("\t\t" + file_path)
    
    def test_preprocess(self):
        sample_rate = 22050
        n_bins = 84
        
        def load_and_preprocess(fn, tid):
            wave = load_audio(fn, sample_rate)
            spectrogram = constant_quality_transform(wave, sample_rate, n_bins=n_bins)
            spectrogram = np.expand_dims(spectrogram, -1)
            return spectrogram, tid
        
        dataset = scan_dataset(self.dataset_dir)
        classes, dataset = flatten_dataset(dataset)
        dataset = compile_dataset(dataset, load_and_preprocess, (tf.float32, tf.int32), ([n_bins, None, 1], []))
        
        for i, (spectrogram, class_id) in enumerate(dataset):
            plt.imshow(spectrogram[..., 0])
            plt.show()
            if i >= 5:
                break
    
    def test_random_clip(self):
        sample_rate = 22050
        n_bins = 84
        clip_size = 258
        
        def load_and_preprocess(fn, tid):
            wave = load_audio(fn, sample_rate)
            spectrogram = constant_quality_transform(wave, sample_rate, n_bins=n_bins)
            spectrogram = np.expand_dims(spectrogram, -1)
            return spectrogram, tid
        
        dataset = scan_dataset(self.dataset_dir)
        classes, dataset = flatten_dataset(dataset)
        dataset = compile_dataset(dataset, load_and_preprocess, (tf.float32, tf.int32), ([n_bins, None, 1], []))
        
        def random_clip(sp, tid):
            head = tf.random.uniform([], 0, tf.shape(sp)[1] - clip_size, tf.int32)
            tail = head + clip_size
            sp = sp[:, head:tail, :]
            return sp, tid
        
        dataset = dataset.map(random_clip)
        
        for i, (spectrogram, class_id) in enumerate(dataset):
            plt.imshow(spectrogram[..., 0])
            plt.show()
            if i >= 5:
                break
