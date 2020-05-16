import numpy as np
import tensorflow as tf

from dataset.audio import load_audio
from dataset.dataset import compile_dataset, flatten_dataset, scan_dataset, split_dataset
from dataset.spectrogram import cqt_spectrogram
from model.resnet import Resnet34_v020

# configurations
dataset_dir = "./data"
sample_rate = 22050

clip_size = 430
clip_wave_size = 220000

n_sp = 84
input_shape = [n_sp, clip_size, 1]

# prepare dataset
dataset = scan_dataset(dataset_dir)
train_dataset, test_dataset, _ = split_dataset(dataset)
classes, train_dataset = flatten_dataset(train_dataset)
_, test_dataset = flatten_dataset(test_dataset)


def load_and_preprocess(fn, tid):
    wave = load_audio(fn, sample_rate)
    
    # random clip
    clip_head = np.random.randint(0, len(wave) - clip_wave_size)
    clip_tail = clip_head + clip_wave_size
    wave = wave[clip_head:clip_tail]
    
    spectrogram = cqt_spectrogram(wave, sample_rate, n_cqt=n_sp)
    spectrogram = np.expand_dims(spectrogram, -1)
    return spectrogram, tid


train_dataset = compile_dataset(train_dataset, load_and_preprocess, (tf.float32, tf.int32), (input_shape, []))
test_dataset = compile_dataset(test_dataset, load_and_preprocess, (tf.float32, tf.int32), (input_shape, []))

# prepare model
model = Resnet34_v020(input_shape, len(classes))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'])

# train model
model.fit(
    x=train_dataset.batch(10),
    validation_data=test_dataset.batch(10),
    epochs=100,
    callbacks=[
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='logs/weights_epoch{epoch:04d}.hdf5',
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.TensorBoard(),
    ])
