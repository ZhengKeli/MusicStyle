import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset.extracting import load_extracted_feature
from dataset.splitting import load_ref_files
from model.dense import DeepDenseNet

# conf
sample_rate = 22050
dataset_dir = "./data"

# ref files
class_names, train_set, test_set, valid_set = load_ref_files(dataset_dir)

feature_names = [
    'chroma_stft', 'root_mean_square', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
    'zero_crossing_rate', 'mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8',
    'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19']


# load dataset
def load_subset(subset):
    xs = []
    ys = []
    for audio_filename, class_id in subset:
        x = [load_extracted_feature(audio_filename, feature_name)
             for feature_name in feature_names]
        y = class_id
        xs.append(x)
        ys.append(y)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    return xs, ys


x_train, y_train = load_subset(train_set)
x_test, y_test = load_subset(test_set)
x_valid, y_valid = load_subset(valid_set)

# mix all
# x_all = np.concatenate([x_train, x_test, x_valid], 0)
# y_all = np.concatenate([y_train, y_test, y_valid], 0)
# x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.4)
# x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5)

# mix train and test
x_all = np.concatenate([x_train, x_test], 0)
y_all = np.concatenate([y_train, y_test], 0)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.4)


# normalizing
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_valid = scaler.transform(x_valid)

# prepare model
model = DeepDenseNet(x_train.shape[-1], len(class_names))
model.compile(
    optimizer=tf.keras.optimizers.Adam(2e-4),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'])

# train model
model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test, y_test),
    shuffle=True,
    batch_size=128,
    epochs=400,
    callbacks=[
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(patience=100),
        # tf.keras.callbacks.ModelCheckpoint(
        #     filepath='logs/weights_epoch{epoch:04d}.hdf5',
        #     save_best_only=True,
        #     save_weights_only=True
        # ),
        # tf.keras.callbacks.TensorBoard(),
    ])


# validation
def compute_accuracy(xs, ys):
    p_pred = model.predict(xs)
    y_pred = np.argmax(p_pred, -1)
    accuracy = np.mean(y_pred == ys)
    return accuracy


print("train accuracy =", compute_accuracy(x_train, y_train))
print("test accuracy =", compute_accuracy(x_test, y_test))
print("valid accuracy =", compute_accuracy(x_valid, y_valid))

cm = confusion_matrix(y_valid, np.argmax(model.predict(x_valid), -1))
sns.heatmap(cm, annot=True, vmin=0, vmax=20, xticklabels=class_names, yticklabels=class_names)
plt.tight_layout()
plt.show()
