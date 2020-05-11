import numpy as np
import matplotlib.pyplot as plt
import pyaudio

sr = 44100

ts = np.arange(0, 5 * sr) / sr
ws = (np.sin(ts * 2 * np.pi * 599) + np.sin(ts * 2 * np.pi * 601)) / 3

plt.plot(ts, ws)
plt.show()

ws = np.array(ws * 2 ** 15, np.int16)
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sr, output=True)
stream.write(ws.tobytes())
stream.stop_stream()
stream.close()
audio.terminate()
