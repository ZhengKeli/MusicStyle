import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset.extracting import load_extracted_feature
from dataset.splitting import load_ref_files
from model.fenet import FeNet

# conf
sample_rate = 22050
dataset_dir = "./data"
weights_filename = r"logs_fe\weights_epoch0011.hdf5"

# ref files
class_names, _, _, valid_set = load_ref_files(dataset_dir)

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


x_valid, y_valid = load_subset(valid_set)

# normalizing
mean = np.asarray([
    3.79953772e-01, 1.35098633e-01, 2.21223256e+03, 2.25435209e+03, 4.59181625e+03, 1.03791986e-01,
    -1.40505301e+02, 9.88738169e+01, -8.42032207e+00, 3.61049800e+01, -6.43721419e-01, 1.43071774e+01,
    -4.57101296e+00, 9.85329372e+00, -6.78372848e+00, 7.52090111e+00, -5.85764400e+00, 4.27417758e+00,
    -4.77176053e+00, 1.61790875e+00, -3.76422384e+00, 1.09862577e+00, -3.82369773e+00, 3.08810440e-01,
    -2.38044834e+00, -1.13741981e+00
])
var = np.asarray([
    6.92317562e-03, 4.34771479e-03, 5.15629274e+05, 2.77090815e+05, 2.49724418e+06, 1.76530578e-03,
    1.01135140e+04, 1.02001342e+03, 4.57252820e+02, 2.92882968e+02, 1.47138172e+02, 1.44537688e+02,
    9.44507215e+01, 1.11549897e+02, 6.86688760e+01, 6.17602512e+01, 4.67590099e+01, 4.37835459e+01,
    3.76796875e+01, 2.42575717e+01, 2.24316394e+01, 2.02885014e+01, 1.98705775e+01, 1.54993774e+01,
    1.44679944e+01, 1.47571104e+01
])

x_valid -= mean
x_valid /= np.sqrt(var)

# prepare model
model = FeNet(x_valid.shape[-1], len(class_names))
model.load_weights(weights_filename, by_name=True)


# validation
def compute_accuracy(xs, ys):
    p_pred = model.predict(xs)
    y_pred = np.argmax(p_pred, -1)
    accuracy = np.mean(y_pred == ys)
    return accuracy


print("valid accuracy =", compute_accuracy(x_valid, y_valid))

cm = confusion_matrix(y_valid, np.argmax(model.predict(x_valid), -1))
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, vmin=0, vmax=20,
            xticklabels=class_names, yticklabels=class_names,
            cmap="Blues", cbar=False)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()
