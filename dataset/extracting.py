import os

import numpy as np


def extracted_feature_filename(audio_filename, feature_name):
    return audio_filename + "." + feature_name + ".npy"


def save_extracted_feature(audio_filename, feature_name, feature_value, overwrite=False):
    feature_filename = extracted_feature_filename(audio_filename, feature_name)
    
    if os.path.exists(feature_filename):
        if overwrite:
            os.remove(feature_filename)
        else:
            raise FileExistsError(feature_filename)
    
    np.save(
        file=feature_filename,
        arr=feature_value,
        allow_pickle=False)


def load_extracted_feature(audio_filename, feature_name):
    return np.load(file=extracted_feature_filename(audio_filename, feature_name))
