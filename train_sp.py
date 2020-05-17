import numpy as np
import tensorflow as tf

from dataset.extracting import load_extracted_feature
from dataset.splitting import load_ref_files
from model.resnet import Resnet34_v021, Resnet34_v022

# configurations
dataset_dir = "./data"
sample_rate = 22050

n_sp = 84
clip_size = 1920 // 8

# ref files
class_names, train_set, test_set, _ = load_ref_files(dataset_dir)
spectrogram_name = 'cqt_spectrogram'


# load dataset
def load_subset(subset, noise=0.0):
    for audio_filename, class_id in subset:
        sp = load_extracted_feature(audio_filename, spectrogram_name)
        sp_length = np.shape(sp)[1]
        if clip_size < sp_length:
            clip_head = np.random.randint(0, sp_length - clip_size)
            clip_tail = clip_head + clip_size
            sp = sp[:, clip_head:clip_tail]
        sp = np.expand_dims(sp, -1)
        if noise != 0.0:
            sp += np.random.normal(0, noise, np.shape(sp))
        yield sp, class_id


train_tf_dataset = tf.data.Dataset.from_generator(
    lambda: load_subset(train_set, 0.1),
    (tf.float32, tf.int32), ([n_sp, clip_size, 1], []))

test_tf_dataset = tf.data.Dataset.from_generator(
    lambda: load_subset(test_set),
    (tf.float32, tf.int32), ([n_sp, clip_size, 1], []))

# prepare model
model = Resnet34_v022([n_sp, clip_size, 1], len(class_names))
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
        tf.keras.callbacks.ModelCheckpoint(
            filepath='logs/weights_epoch{epoch:04d}.hdf5',
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.TensorBoard(),
    ])
