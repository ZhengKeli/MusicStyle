import os

import numpy as np


def scan_dataset(dataset_dir):
    if not os.path.isdir(dataset_dir):
        raise ValueError("Path " + dataset_dir + " is not a directory!")
    
    dataset = {}
    for type_name in os.listdir(dataset_dir):
        type_dir = os.path.join(dataset_dir, type_name)
        if not os.path.isdir(type_dir):
            continue
        
        filename_list = []
        for file_name in os.listdir(type_dir):
            file_path = os.path.join(type_dir, file_name)
            filename_list.append(file_path)
        
        if filename_list:
            dataset[type_name] = tuple(filename_list)
    
    return dataset


def map_dataset(dataset, func):
    new_dataset = {}
    for type_name, item_list in dataset.items():
        new_item_list = tuple(func(item) for item in item_list)
        new_dataset[type_name] = new_item_list
    return new_dataset


def split_dataset(dataset: dict, ratios=(3, 1, 1), shuffle=False):
    """ Split the dataset into several parts
    
    :param dataset: The dataset to be split.
    :param ratios: ratios of quantity
    :param shuffle: if shuffle before splitting
    :return: a tuple of several split dataset
    """
    ratios = np.array(ratios, np.float)
    ratios = np.cumsum(ratios)
    ratios /= np.sum(ratios[-1])
    
    subsets = tuple({} for _ in ratios)
    for type_name, file_list in dataset.items():
        if shuffle:
            file_list = np.array(file_list)
            np.random.shuffle(file_list)
        tails = np.asarray(ratios * len(file_list), np.int)
        head = 0
        for subset, tail in zip(subsets, tails):
            subset[type_name] = tuple(file_list[head:tail])
            head = tail
    
    return subsets
