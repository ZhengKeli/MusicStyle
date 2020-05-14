import numpy as np
import tensorflow as tf

from dataset.audio import load_audio
from dataset.dataset import compile_dataset, flatten_dataset, scan_dataset, split_dataset
from dataset.preprocess import constant_quality_transform
from model.resnet import Resnet34_modified

# configurations
dataset_dir = "./data"
sample_rate = 22050

clip_size = 258
n_bins = 84

# prepare dataset
dataset = scan_dataset(dataset_dir)
train_dataset, test_dataset, _ = split_dataset(dataset)
classes, train_dataset = flatten_dataset(train_dataset)
_, test_dataset = flatten_dataset(test_dataset)


def load_and_preprocess(fn, tid):
    wave = load_audio(fn, sample_rate)
    spectrogram = constant_quality_transform(wave, sample_rate, n_bins=n_bins)
    spectrogram = np.expand_dims(spectrogram, -1)
    return spectrogram, tid


train_dataset = compile_dataset(train_dataset, load_and_preprocess, (tf.float32, tf.int32), ([n_bins, None, 1], []))
test_dataset = compile_dataset(test_dataset, load_and_preprocess, (tf.float32, tf.int32), ([n_bins, None, 1], []))


def random_clip(sp, tid):
    head = tf.random.uniform([], 0, tf.shape(sp)[1] - clip_size, tf.int32)
    tail = head + clip_size
    sp = sp[:, head:tail, :]
    return sp, tid


train_dataset = train_dataset.map(random_clip)
test_dataset = test_dataset.map(random_clip)

# prepare model
model = Resnet34_modified((n_bins, clip_size, 1), len(classes))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'])

# train model
model.fit(
    x=train_dataset.batch(5),
    validation_data=test_dataset.batch(5),
    epochs=200,
    callbacks=[
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(patience=50),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='logs/weights_epoch{epoch:04d}.hdf5',
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.TensorBoard(),
    ])
