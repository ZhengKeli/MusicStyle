import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

sr = 44100

ts = np.arange(0, 5 * sr) / sr
ws = (np.sin(ts * 2 * np.pi * 599) + np.sin(ts * 2 * np.pi * 601)) / 3

plt.plot(ts, ws)
plt.show()

ws = np.array(ws * 2 ** 15, np.int16)
segment = AudioSegment(ws.tobytes(), sample_width=2, channels=1, frame_rate=sr)
segment.export("./beat.mp3", 'mp3')
