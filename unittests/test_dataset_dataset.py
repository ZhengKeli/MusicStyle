import unittest

from dataset import load_raw_dataset


class TestDatasetDataset(unittest.TestCase):
    def setUp(self):
        self.datset_dir = r"C:\Users\keli\OneDrive\学习\CIS Machine Learning\MusicStyle\data\genres"
    
    def test_load_dataset(self):
        dataset = load_raw_dataset(self.datset_dir)
        self.assertTrue(len(dataset.keys()) == 10)
        for type_name, file_list in dataset.items():
            print(type_name + ":")
            self.assertTrue(len(file_list) == 100)
            for file_path in file_list:
                print("\t" + file_path)
