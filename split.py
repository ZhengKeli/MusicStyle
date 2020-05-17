import numpy as np
import os
from dataset.splitting import flatten_dataset, save_cls_file, save_ref_file, scan_dataset, split_dataset

# configurations
dataset_dir = ".\\data"
overwrite = True

# scan
dataset = scan_dataset(dataset_dir)
print('scanned dataset')

train_set, test_set, valid_set = split_dataset(dataset, (3, 1, 1))
classes, train_set = flatten_dataset(train_set)
_, test_set = flatten_dataset(test_set)
_, valid_set = flatten_dataset(valid_set)
print('split dataset')

# save
classes_fn = os.path.join(dataset_dir, 'classes.txt')
save_cls_file(classes, classes_fn)
print('saved class file to', classes_fn)

train_set_fn = os.path.join(dataset_dir, "train_set.txt")
save_ref_file(train_set, train_set_fn, overwrite=overwrite)
print('saved train set ref file to', train_set_fn)

test_set_fn = os.path.join(dataset_dir, "test_set.txt")
save_ref_file(test_set, test_set_fn, overwrite=overwrite)
print('saved test set ref file to', test_set_fn)

valid_set_fn = os.path.join(dataset_dir, "valid_set.txt")
save_ref_file(valid_set, valid_set_fn, overwrite=overwrite)
print('saved valid set ref file to', valid_set_fn)
