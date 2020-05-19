import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataset.extracting import load_extracted_feature
from dataset.splitting import load_ref_files
from model.resnet import Resnet022

# configurations
dataset_dir = "./data"
sample_rate = 22050
weights_filename = r"logs_archive\logs_resnet022_mfcc\weights_epoch0038.hdf5"

n_sp = 84
clip_size = 1290  # 1290

# ref files
class_names, _, test_set, valid_set = load_ref_files(dataset_dir)

spectrogram_name = 'mfcc_spectrogram'


# load dataset
def load_subset(subset):
    xs = []
    ys = []
    for audio_filename, class_id in subset:
        sp = load_extracted_feature(audio_filename, spectrogram_name)
        sp_length = np.shape(sp)[1]
        if clip_size < sp_length:
            clip_head = np.random.randint(0, sp_length - clip_size)
            clip_tail = clip_head + clip_size
            sp = sp[:, clip_head:clip_tail]
        sp = np.expand_dims(sp, -1)
        xs.append(sp)
        
        y = class_id
        ys.append(y)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    return xs, ys


x_test, y_test = load_subset(test_set)
x_valid, y_valid = load_subset(valid_set)

# prepare model
model = Resnet022([n_sp, clip_size, 1], len(class_names))
model.load_weights(weights_filename, by_name=True)

# perform the validation
prob_pred = model.predict(x_valid)
y_pred = np.argmax(prob_pred, -1)
valid_accuracy = np.mean(y_valid == y_pred)
print("valid accuracy =", valid_accuracy)

sns.set()
cm = confusion_matrix(y_valid, y_pred)
sns.heatmap(cm, annot=True, vmin=0, vmax=20, xticklabels=class_names, yticklabels=class_names)
plt.tight_layout()
plt.show()
