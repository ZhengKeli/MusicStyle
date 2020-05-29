import librosa
import numpy as np

from dataset.audio import play_audio
from dataset.spectrogram import cqt_spectrogram, mfcc_spectrogram
from model.spnet import SpNet

# configurations
audio_filename = r".\Scarborough Fair.mp3"
sample_rate = 22050
clip_duration = 30  # load only part of audio (in seconds)
# clip_duration = None # load the whole audio (it may take a long time)

weights_filename = r"logs_sp/weights_epoch0190.hdf5"

n_cqt = 84
n_mfcc = 84

class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
input_shape = ([n_cqt, None, 1], [n_mfcc, None, 1])

# load and preprocess
file_duration = librosa.get_duration(filename=audio_filename)
if clip_duration is not None:
    offset = np.random.uniform(0, file_duration - clip_duration)
    print(f"will load only {clip_duration:.1f} s from the audio ({offset:.1f} - {offset + clip_duration:.1f} s).")
    
    print(f"loading audio file {audio_filename}")
    wave, _ = librosa.load(audio_filename, sample_rate, offset=offset, duration=float(clip_duration))
else:
    print(f"will load the whole audio (totally {file_duration:.1f} s).")
    
    print(f"loading audio file {audio_filename}")
    wave, _ = librosa.load(audio_filename, sample_rate)

print('computing cqt spectrogram')
cqt_sp = cqt_spectrogram(wave, sample_rate, n_cqt)

print('computing mfcc spectrogram')
mfcc_sp = mfcc_spectrogram(wave, sample_rate, n_mfcc)

print('clipping and reshaping')
sps = [cqt_sp, mfcc_sp]
sps = [np.expand_dims(sp, -1) for sp in sps]
sps = [np.expand_dims(sp, 0) for sp in sps]

# prepare model
print('preparing model')
model = SpNet([n_cqt, None, 1], [n_mfcc, None, 1], len(class_names))
model.load_weights(weights_filename, by_name=True)

print('performing prediction')
prob_pred = model.predict(sps)[0]
y_pred = int(np.argmax(prob_pred, -1))

print(f"The prediction of file {audio_filename}")
print('\t' + class_names[y_pred])
print("Probabilities of all classes:")
for prob, class_name in zip(prob_pred, class_names):
    print('\t' + class_name, ":", f'{prob:.2%}')

print("playing loaded piece of music")
play_audio(wave, sample_rate)
