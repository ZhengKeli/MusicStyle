import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataset.extracting import load_extracted_feature
from dataset.splitting import load_ref_files
from model.spnet import SpNet_v1

# configurations
dataset_dir = "./data"
sample_rate = 22050
weights_filename = r"logs_archive\logs_spnet1_2\weights_epoch0087.hdf5"

n_sp = 84
clip_size = 1290  # 1290

# ref files
class_names, _, test_set, valid_set = load_ref_files(dataset_dir)
spectrogram_names = ['cqt_spectrogram', 'mfcc_spectrogram']


# load dataset
def random_clip_spectrogram(spectrogram, clip_size):
    sp_length = np.shape(spectrogram)[1]
    if clip_size < sp_length:
        clip_head = np.random.randint(0, sp_length - clip_size)
        clip_tail = clip_head + clip_size
        spectrogram = spectrogram[:, clip_head:clip_tail]
    return spectrogram


def load_subset(subset, noise=0.0):
    sp1s = []
    sp2s = []
    ys = []
    for audio_filename, class_id in subset:
        sps = []
        for spectrogram_name in spectrogram_names:
            sp = load_extracted_feature(audio_filename, spectrogram_name)
            sp = random_clip_spectrogram(sp, clip_size)
            sp = np.expand_dims(sp, -1)
            if noise != 0.0:
                sp += np.random.normal(0, noise, np.shape(sp))
            sps.append(sp)
        sp1, sp2 = sps
        sp1s.append(sp1)
        sp2s.append(sp2)
        ys.append(class_id)
    sp1s = np.asarray(sp1s)
    sp2s = np.asarray(sp2s)
    ys = np.asarray(ys)
    return (sp1s, sp2s), ys


x_test, y_test = load_subset(test_set)
x_valid, y_valid = load_subset(valid_set)

# prepare model
model = SpNet_v1([n_sp, clip_size, 1], [n_sp, clip_size, 1], len(class_names))
model.load_weights(weights_filename, by_name=True)

# on test set
# perform the validation
prob_pred = model.predict(x_test)
y_pred = np.argmax(prob_pred, -1)
test_accuracy = np.mean(y_test == y_pred)
print("test accuracy =", test_accuracy)

sns.set()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, vmin=0, vmax=20, xticklabels=class_names, yticklabels=class_names)
plt.tight_layout()
plt.show()

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
