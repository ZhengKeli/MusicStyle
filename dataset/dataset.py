import os
import numpy as np


def load_audio_dataset(dataset_dir, extension='wav'):
    if not os.path.isdir(dataset_dir):
        raise ValueError("Path " + dataset_dir + " is not a directory!")
    
    dataset = {}
    for type_name in os.listdir(dataset_dir):
        type_dir = os.path.join(dataset_dir, type_name)
        if not os.path.isdir(type_dir):
            continue
        
        file_list = []
        for file_name in os.listdir(type_dir):
            if not file_name.endswith('.' + extension):
                continue
            file_path = os.path.join(type_dir, file_name)
            file_list.append(file_path)
        
        if file_list:
            dataset[type_name] = tuple(file_list)
    
    return dataset


def split_dataset(dataset: dict, ratios=(3, 1, 1)):
    """ Split the dataset into several parts
    
    :param dataset: The dataset to be split.
    :param ratios: ratios of quantity
    :return: a tuple of several split dataset
    """
    ratios = np.array(ratios, np.float)
    ratios = np.cumsum(ratios)
    ratios /= np.sum(ratios[-1])
    
    subsets = tuple({} for _ in ratios)
    for type_name, file_list in dataset.items():
        tails = np.asarray(ratios * len(file_list), np.int)
        head = 0
        for subset, tail in zip(subsets, tails):
            subset[type_name] = tuple(file_list[head:tail])
            head = tail
    
    return subsets
