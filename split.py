import numpy as np
import os
from dataset.dataset import flatten_dataset, scan_dataset, split_dataset
from dataset.splitting import save_cls_file, save_ref_file

# configurations
seed = np.random.randint(100000)
dataset_dir = ".\\data"
np.random.seed(seed)

# scan
dataset = scan_dataset(dataset_dir)
train_set, test_set, valid_set = split_dataset(dataset, (3, 1, 1))

classes, train_set = flatten_dataset(train_set)
_, test_set = flatten_dataset(test_set)
_, valid_set = flatten_dataset(valid_set)

# save
classes_fn = os.path.join(dataset_dir, 'classes.txt')
save_cls_file(classes, classes_fn)

train_set_fn = os.path.join(dataset_dir, "train_set.txt")
save_ref_file(train_set, train_set_fn)

test_set_fn = os.path.join(dataset_dir, "test_set.txt")
save_ref_file(test_set, test_set_fn)

valid_set_fn = os.path.join(dataset_dir, "valid_set.txt")
save_ref_file(valid_set, valid_set_fn)
