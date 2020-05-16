import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from dataset.audio import load_audio
from dataset.dataset import compile_dataset, flatten_dataset, scan_dataset, split_dataset
from dataset.spectrogram import cqt_spectrogram
from model.resnet import Resnet34_v011, Resnet34_v020

# configurations
dataset_dir = "./data"
sample_rate = 22050
weights_filename = "logs_archive\logs_resnet011\weights_epoch0019.hdf5"

clip_size = 258
clip_wave_size = (clip_size - 1) * 512
n_sp = 84
input_shape = [n_sp, clip_size, 1]

# prepare dataset
dataset = scan_dataset(dataset_dir)
_, valid_dataset,_,  = split_dataset(dataset)
classes, valid_dataset = flatten_dataset(valid_dataset)


def load_and_preprocess(fn, tid):
    wave = load_audio(fn, sample_rate)
    
    # random clip
    clip_head = np.random.randint(0, len(wave) - clip_wave_size)
    clip_tail = clip_head + clip_wave_size
    wave = wave[clip_head:clip_tail]
    
    spectrogram = cqt_spectrogram(wave, sample_rate, n_cqt=n_sp, norm=False)
    spectrogram = np.expand_dims(spectrogram, -1)
    return spectrogram, tid


valid_dataset = compile_dataset(valid_dataset, load_and_preprocess, (tf.float32, tf.int32), (input_shape, []))

# prepare model
model = Resnet34_v011(input_shape, len(classes))
model.load_weights(weights_filename, by_name=True)

# perform the validation
valid_y = []
pred_y = []
for batch_x, batch_y in valid_dataset.batch(5).as_numpy_iterator():
    batch_pred_prob = model.predict(batch_x)
    batch_pred_y = np.argmax(batch_pred_prob, -1)
    valid_y.append(batch_y)
    pred_y.append(batch_pred_y)

valid_y = np.concatenate(valid_y)
pred_y = np.concatenate(pred_y)

# show result
accuracy = np.mean(valid_y == pred_y)
print("accuracy =", accuracy)

sns.set()
cm = confusion_matrix(valid_y, pred_y)
sns.heatmap(cm, annot=True, vmin=0, vmax=20, xticklabels=classes, yticklabels=classes)
plt.tight_layout()
plt.show()
