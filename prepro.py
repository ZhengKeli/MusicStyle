import numpy as np
from dataset import load_audio, map_dataset, scan_dataset
from preprocess.preprocess import constant_quality_transform

# configurations
sample_rate = 22050
dataset_dir = r"C:\Users\keli\OneDrive\学习\CIS Machine Learning\MusicStyle\data\genres"
output_filename = "./dataset.npz"

# scan dataset
dataset = scan_dataset(dataset_dir)
print("scanned audio files")

# load and preprocess
print("load and performing the preprocess")
preprocess = lambda w: np.expand_dims(constant_quality_transform(w, sample_rate, 512), -1)
dataset = map_dataset(dataset, lambda fn: preprocess(load_audio(fn, sample_rate)))
print("finished preprocess")

# clip
length = np.min([
    [np.shape(item)[1] for item in item_list]
    for item_list in dataset.values()
])
width = np.min([
    [np.shape(item)[0] for item in item_list]
    for item_list in dataset.values()
])
for type_name, item_list in dataset.items():
    clipped_item_list = [item[:width, :length] for item in item_list]
    dataset[type_name] = np.stack(clipped_item_list)
print("clipped img data")

# save
print("saving the preprocessed dataset to file ", output_filename)
np.savez_compressed(output_filename, **dataset)
print("file saved")
