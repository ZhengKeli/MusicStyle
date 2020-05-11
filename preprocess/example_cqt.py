import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from dataset.audio import load_audio

sample_rate = 22050
mp3_filename = r"C:\Users\keli\OneDrive\学习\CIS Machine Learning\MusicStyle\data\genres\blues\blues.00000.wav"

wave = load_audio(mp3_filename, sample_rate)
plt.plot(wave)
plt.show()

cqt = librosa.cqt(wave, sample_rate)
cqt = librosa.amplitude_to_db(cqt, ref=np.max)

plt.figure(figsize=(10, 3))
librosa.display.specshow(cqt, y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrogram (note)')
plt.show()
