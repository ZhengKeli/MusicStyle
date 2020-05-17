import os
import numpy as np

from dataset.extracting import load_extracted_feature
from dataset.splitting import load_ref_files

# conf
sample_rate = 22050
dataset_dir = "./data"

feature_names = [
    'chroma_stft', 'root_mean_square', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
    'zero_crossing_rate', 'mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8',
    'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19']

spectrogram_names = ['mfcc_spectrogram', 'cqt_spectrogram']

# ref files
class_names, train_set, test_set, valid_set = load_ref_files(dataset_dir)

# load dataset
subsets = {'train_set': train_set, 'test_set': test_set, 'valid_set': valid_set}
for subset_name, subset in subsets.items():
    for audio_filename, class_id in subset:
        class_name = class_names[class_id]
        print(os.path.basename(audio_filename))
        for feature_name in feature_names:
            feature_value = load_extracted_feature(audio_filename, feature_name)
            print('\t', feature_name, '=', feature_value)
        for spectrogram_name in spectrogram_names:
            spectrogram = load_extracted_feature(audio_filename, spectrogram_name)
            print('\t', spectrogram_name + '.shape =', np.shape(spectrogram))
