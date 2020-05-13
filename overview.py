import numpy as np
import matplotlib.pyplot as plt

# configurations
sample_rate = 22050
dataset_filename = "./dataset.npz"

# load dataset
print("loading dataset from file " + dataset_filename)
dataset = np.load(dataset_filename)
print("dataset loaded")

# show
fig, axes = plt.subplots(10, 3, figsize=(10, 7))
for (tn, il), ax in zip(dataset.items(), axes):
    for im, a in zip(il, ax):
        a.imshow(im)
        a.axis('off')
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()
