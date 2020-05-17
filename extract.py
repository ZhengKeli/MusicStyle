import librosa
import numpy as np

from dataset.audio import load_audio
from dataset.dataset import scan_dataset
from dataset.extracting import save_extracted_feature
from dataset.spectrogram import cqt_spectrogram, mfcc_spectrogram

# conf

dataset_dir = "./data"
sample_rate = 22050


# extractors

def wave_features_extractor(wave):
    chroma_stft = librosa.feature.chroma_stft(wave, sample_rate)
    chroma_stft = np.mean(chroma_stft)
    
    root_mean_square = librosa.feature.rms(wave)
    root_mean_square = np.mean(root_mean_square)
    
    spectral_centroid = librosa.feature.spectral_centroid(wave, sample_rate)
    spectral_centroid = np.mean(spectral_centroid)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(wave, sample_rate)
    spectral_bandwidth = np.mean(spectral_bandwidth)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(wave, sample_rate)
    spectral_rolloff = np.mean(spectral_rolloff)
    
    zero_crossing_rate = librosa.feature.zero_crossing_rate(wave)
    zero_crossing_rate = np.mean(zero_crossing_rate)
    
    return dict(
        chroma_stft=chroma_stft,
        root_mean_square=root_mean_square,
        spectral_centroid=spectral_centroid,
        spectral_bandwidth=spectral_bandwidth,
        spectral_rolloff=spectral_rolloff,
        zero_crossing_rate=zero_crossing_rate
    )


def mfcc_features_extractor(wave):
    mfcc = librosa.feature.mfcc(wave, sample_rate)
    mfcc = np.mean(mfcc, -1)
    return dict(('mfcc' + str(i), mfcc_i) for i, mfcc_i in enumerate(mfcc))


def spectrogram_extractor(wave):
    mfcc_sp = mfcc_spectrogram(wave, sample_rate, 84)
    cqt_sp = cqt_spectrogram(wave, sample_rate, 84)
    return dict(
        mfcc_spectrogram=mfcc_sp,
        cqt_spectrogram=cqt_sp
    )


# extract

all_extractors = [
    wave_features_extractor,
    mfcc_features_extractor,
    spectrogram_extractor,
]
all_feature_names = None

dataset = scan_dataset(dataset_dir)
for cls, filename_list in dataset.items():
    for filename in filename_list:
        wave = load_audio(filename, sample_rate)
        
        features = {}
        for extractor in all_extractors:
            features = {**features, **extractor(wave)}
        
        if all_feature_names is None:
            all_feature_names = list(features.keys())
            print('feature_names =', all_feature_names)
        
        for feature_name, feature_value in features.items():
            save_extracted_feature(filename, feature_name, feature_value)
        
        print("\tfinished extraction for file " + filename)
    print("finished extraction for class " + cls)
print("finished all")
