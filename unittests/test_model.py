import unittest

from model.resnet import Resnet34_modified


class TestDatasetAudio(unittest.TestCase):
    def test_resnet34_modified(self):
        model = Resnet34_modified((84, 1290, 1), 10)
        print(model)
