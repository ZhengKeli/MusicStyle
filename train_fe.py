import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from model.dense import DeepDenseNet

# reading dataset from csv
dataset = pd.read_csv('data/dataset_2.csv')
dataset = dataset.drop(['filename'], axis=1)

genre_list = dataset.iloc[:, -1]
encoder = LabelEncoder()
y_all = encoder.fit_transform(genre_list)

# normalizing
scaler = StandardScaler()
x_all = scaler.fit_transform(np.array(dataset.iloc[:, :-1], dtype=float))

# splitting of dataset into train and test dataset
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2)

# prepare the model
model = DeepDenseNet(x_train.shape[1], len(encoder.classes_))
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'])

# train model
model.fit(
    x=x_train,
    y=y_train,
    batch_size=128,
    epochs=200,
    validation_data=(x_test, y_test),
    callbacks=[
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(patience=50),
        # tf.keras.callbacks.ModelCheckpoint(
        #     filepath='logs_archive/logs_0.0.1/weights_epoch{epoch:04d}.hdf5',
        #     save_best_only=True,
        #     save_weights_only=True
        # ),
        # tf.keras.callbacks.TensorBoard(),
    ])

pred_prob = model.predict(x_test)
y_pred = np.argmax(pred_prob, -1)

accuracy = np.mean(y_test == y_pred)
print("accuracy =", accuracy)

sns.set()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, vmin=0, vmax=20, xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.tight_layout()
plt.show()
