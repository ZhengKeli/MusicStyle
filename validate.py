import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataset import split_dataset
from dataset.dataset import flatten_dataset
from model.resnet import Resnet34_modified

# configurations
sample_rate = 22050
dataset_filename = "./dataset.npz"
weights_filename = r"logs_archive\logs_0.0.2\weights_epoch0007.hdf5"

# load dataset
print("loading dataset from file " + dataset_filename)
dataset = np.load(dataset_filename)
print("dataset loaded")

# split and flatten
print("splitting and flattening dataset")

_, _, validate_dataset = split_dataset(dataset)
del _, dataset

classes, valid_x, valid_y = flatten_dataset(validate_dataset)
del validate_dataset

print("dataset prepared")

# prepare model
model = Resnet34_modified(np.shape(valid_x)[1:], len(classes))
model.load_weights(weights_filename, by_name=True)

# perform the validation
pred_prob = model.predict(valid_x)
pred_y = np.argmax(pred_prob, -1)

# show result
accuracy = np.mean(valid_y == pred_y)
print("accuracy =", accuracy)

sns.set()
cm = confusion_matrix(valid_y, pred_y)
sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes)
plt.tight_layout()
plt.show()
