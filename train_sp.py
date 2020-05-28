import numpy as np
import tensorflow as tf

from dataset.extracting import load_extracted_feature
from dataset.splitting import load_ref_files
from model.spnet import SpNet

# configurations
dataset_dir = "./data"
sample_rate = 22050

n_cqt = 84
n_mfcc = 84
n_mel = 128
clip_size = 1920 // 4

spectrogram_names = ['cqt_spectrogram', 'mfcc_spectrogram']
input_shape = ([n_cqt, None, 1], [n_mfcc, None, 1])


# load dataset

def random_clip_spectrogram(spectrogram, clip_size):
    sp_length = np.shape(spectrogram)[1]
    if clip_size < sp_length:
        clip_head = np.random.randint(0, sp_length - clip_size)
        clip_tail = clip_head + clip_size
        spectrogram = spectrogram[:, clip_head:clip_tail]
    return spectrogram


def load_subset(subset, noise=0.0):
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
        yield (sp1, sp2), class_id


class_names, train_set, test_set, _ = load_ref_files(dataset_dir)

train_tf_dataset = tf.data.Dataset.from_generator(
    lambda: load_subset(train_set),
    ((tf.float32, tf.float32), tf.int32), (input_shape, []))

test_tf_dataset = tf.data.Dataset.from_generator(
    lambda: load_subset(test_set),
    ((tf.float32, tf.float32), tf.int32), (input_shape, []))

# prepare motdel
model = SpNet(*input_shape, len(class_names))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'])

# train model
model.fit(
    x=train_tf_dataset.batch(8),
    validation_data=test_tf_dataset.batch(8),
    shuffle=True,
    epochs=300,
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
