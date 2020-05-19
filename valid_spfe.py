import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataset.extracting import load_extracted_feature
from dataset.splitting import load_ref_files
from model.resnet import Resnet031

# configurations
dataset_dir = "./data"
sample_rate = 22050
weights_filename = r"logs\weights_epoch0030.hdf5"

n_sp = 84
clip_size = 1290

# ref files
class_names, _, _, valid_set = load_ref_files(dataset_dir)

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


valid_sp = []
valid_fe = []
valid_cli = []
for (sp, fe), cli in load_subset(valid_set):
    valid_sp.append(sp)
    valid_fe.append(fe)
    valid_cli.append(cli)

x_valid = np.asarray(valid_sp), np.asarray(valid_fe)
y_valid = np.asarray(valid_cli)

# prepare model
model = Resnet031([n_sp, clip_size, 1], [len(feature_names)], len(class_names))
model.load_weights(weights_filename, by_name=True)

# train model
prob_pred = model.predict(x_valid)
y_pred = np.argmax(prob_pred, -1)
valid_accuracy = np.mean(y_valid == y_pred)
print("valid accuracy =", valid_accuracy)

sns.set()
cm = confusion_matrix(y_valid, y_pred)
sns.heatmap(cm, annot=True, vmin=0, vmax=20, xticklabels=class_names, yticklabels=class_names)
plt.tight_layout()
plt.show()
