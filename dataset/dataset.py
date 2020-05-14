import os
import tensorflow as tf
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


def split_dataset(dataset: dict, ratios=(3, 1, 1), shuffle=False):
    """ Split the dataset into several parts
    
    :param dataset: The dataset to be split.
    :param ratios: ratios of quantity
    :param shuffle: if shuffle before splitting
    :return: a tuple of several split dataset
    """
    ratios = np.asarray(ratios, np.float)
    ratios = np.cumsum(ratios)
    ratios /= np.sum(ratios[-1])
    
    subsets = tuple({} for _ in ratios)
    for type_name, item_list in dataset.items():
        if shuffle:
            item_list = np.asarray(item_list)
            np.random.shuffle(item_list)
        tails = np.asarray(ratios * len(item_list), np.int)
        head = 0
        for subset, tail in zip(subsets, tails):
            subset[type_name] = tuple(item_list[head:tail])
            head = tail
    
    return subsets


def flatten_dataset(dataset: dict, shuffle=True):
    flattened_dataset = []
    type_names = tuple(dataset.keys())
    for type_id, type_name in enumerate(type_names):
        item_list = dataset[type_name]
        for item in item_list:
            flattened_dataset.append((item, type_id))
    
    if shuffle:
        np.random.shuffle(flattened_dataset)
    
    flattened_dataset = tuple(flattened_dataset)
    return type_names, flattened_dataset


def compile_dataset(dataset, preprocess, output_types, output_shapes):
    def data_gen():
        for item in dataset:
            yield preprocess(*item)
    
    tf_dataset = tf.data.Dataset.from_generator(data_gen, output_types, output_shapes)
    return tf_dataset
