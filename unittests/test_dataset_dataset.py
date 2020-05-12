import unittest

from dataset import scan_dataset
from dataset.dataset import split_dataset


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
