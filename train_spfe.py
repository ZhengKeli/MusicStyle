import numpy as np
import tensorflow as tf

from dataset.extracting import load_extracted_feature
from dataset.splitting import load_ref_files
from model.resnet import Resnet34_v031

# configurations
dataset_dir = "./data"
sample_rate = 22050

n_sp = 84
clip_size = 430

# ref files
class_names, train_set, test_set, _ = load_ref_files(dataset_dir)

feature_names = [
    'chroma_stft', 'root_mean_square', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
    'zero_crossing_rate', 'mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8',
    'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19']
spectrogram_name = 'cqt_spectrogram'


# load dataset
def load_subset(subset):
    for audio_filename, class_id in subset:
        sp = load_extracted_feature(audio_filename, spectrogram_name)
        sp_length = np.shape(sp)[1]
        if clip_size < sp_length:
            clip_head = np.random.randint(0, sp_length - clip_size)
            clip_tail = clip_head + clip_size
            sp = sp[:, clip_head:clip_tail]
        sp = np.expand_dims(sp, -1)
        
        fe = [load_extracted_feature(audio_filename, feature_name) for feature_name in feature_names]
        
        yield (sp, fe), class_id


train_tf_dataset = tf.data.Dataset.from_generator(
    lambda: load_subset(train_set),
    ((tf.float32, tf.float32), tf.int32), (([n_sp, clip_size, 1], [len(feature_names)]), []))

test_tf_dataset = tf.data.Dataset.from_generator(
    lambda: load_subset(test_set),
    ((tf.float32, tf.float32), tf.int32), (([n_sp, clip_size, 1], [len(feature_names)]), []))

# prepare model
model = Resnet34_v031([n_sp, clip_size, 1], [len(feature_names)], len(class_names))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'])

# train model
model.fit(
    x=train_tf_dataset.batch(8),
    validation_data=test_tf_dataset.batch(8),
    epochs=200,
    verbose=2,
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
