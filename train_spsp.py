import numpy as np
import tensorflow as tf

from dataset.extracting import load_extracted_feature
from dataset.splitting import load_ref_files
from model.resnet import Resnet040, Resnet041

# configurations
dataset_dir = "./data"
sample_rate = 22050

n_cqt = 84
n_mfcc = 84
clip_size = 1920 // 6

input_shape = ([n_cqt, None, 1], [n_mfcc, None, 1])

# ref files
class_names, train_set, test_set, _ = load_ref_files(dataset_dir)

# class_names, train_set1, train_set2, test_set = load_ref_files(dataset_dir)
# train_set = [*train_set1, *train_set2]

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


train_tf_dataset = tf.data.Dataset.from_generator(
    lambda: load_subset(train_set, 0.1),
    ((tf.float32, tf.float32), tf.int32), (input_shape, []))

test_tf_dataset = tf.data.Dataset.from_generator(
    lambda: load_subset(test_set),
    ((tf.float32, tf.float32), tf.int32), (input_shape, []))

# prepare model
model = Resnet041(*input_shape, len(class_names))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'])

# train model
model.fit(
    x=train_tf_dataset.batch(8),
    validation_data=test_tf_dataset.batch(8),
    shuffle=True,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(patience=30),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='logs/weights_epoch{epoch:04d}.hdf5',
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.TensorBoard(),
    ])
