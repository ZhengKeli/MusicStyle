import numpy as np
import tensorflow as tf
from dataset.dataset import flatten_dataset, split_dataset
from model.resnet import Resnet34

# configurations
sample_rate = 22050
dataset_filename = "./dataset.npz"

# load dataset
print("loading dataset from file " + dataset_filename)
dataset = np.load(dataset_filename)
print("dataset loaded")

# split and flatten
print("splitting and flattening dataset")

train_dataset, test_dataset, _ = split_dataset(dataset)
del dataset

classes, train_x, train_y = flatten_dataset(train_dataset)
del train_dataset

_, test_x, test_y = flatten_dataset(test_dataset)
del test_dataset

print("dataset split and flattened")

# prepare model
model = Resnet34(np.shape(train_x)[1:], len(classes))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

# train model
model.fit(
    x=train_x,
    y=train_y,
    batch_size=8,
    epochs=100,
    validation_data=(test_x, test_y),
    callbacks=[
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(patience=50),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./logs/weights_epoch{epoch:04d}.hdf5',
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.TensorBoard(),
    ]
)
