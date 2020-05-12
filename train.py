from dataset import load_audio, scan_dataset
from dataset.dataset import map_dataset, split_dataset
from preprocess.preprocess import constant_quality_transform

# configurations
sample_rate = 22050
dataset_dir = r"C:\Users\keli\OneDrive\学习\CIS Machine Learning\MusicStyle\data\mini-genres"

# scan dataset
dataset = scan_dataset(dataset_dir)

# load and preprocess
preprocess = lambda w: constant_quality_transform(w, sample_rate, 512)
dataset = map_dataset(dataset, lambda fn: preprocess(load_audio(fn, sample_rate)))

# split
train_dataset, test_dataset, validation_dataset = split_dataset(dataset)

# model
# todo build the model

# train
# todo train the model
