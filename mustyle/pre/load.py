import librosa.display
import matplotlib.pyplot as plt
import numpy as np

mp3_filename = "../../music.mp3"
ws, sr = librosa.load(mp3_filename, sr=None)

cqt = librosa.cqt(ws, sr)
cqt = librosa.amplitude_to_db(cqt, ref=np.max)

plt.figure(figsize=(10, 3))
librosa.display.specshow(cqt, y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrogram (note)')
