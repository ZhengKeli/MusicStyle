import os
import librosa
import pyaudio
import pydub
import numpy as np

from .utils import convert_dtype


def load_audio(filename, sample_rate=22050, dtype=np.float32) -> np.ndarray:
    """ Load an audio file'
    
    :param filename: the filename of the audio file.
    :param sample_rate: the target sample_rate.
    :param dtype: data type of the output array
    :return: an array with shape [sample_count]
    """
    
    dtype = np.dtype(dtype)
    wave, _ = librosa.load(filename, sr=sample_rate, mono=True, dtype=dtype)
    return wave


def save_audio(array, filename, sample_rate, dtype=np.int16, format=None):
    dtype = np.dtype(dtype)
    allowed_dtypes = [np.int8, np.int16, np.int32, np.int64]
    if dtype not in allowed_dtypes:
        raise TypeError("The dtype must be one of " + str(allowed_dtypes))
    
    if np.ndim(array) != 1:
        raise TypeError("Saving multi-channel audio is not supported!")
    
    if format is None:
        name = os.path.basename(filename)
        ext = name.rfind('.')
        if ext == -1:
            raise ValueError("Can not infer output format from the filename!")
        format = name[ext + 1:]
    
    array = convert_dtype(array, dtype)
    
    segment = pydub.AudioSegment(array.tobytes(), sample_width=dtype.itemsize, channels=1, frame_rate=sample_rate)
    segment.export(filename, format)


def play_audio(array, sample_rate, dtype=np.int16):
    dtype = np.dtype(dtype)
    allowed_dtypes = [np.int8, np.int16, np.int32, np.float32]
    allowed_formats = [pyaudio.paInt8, pyaudio.paInt16, pyaudio.paInt32, pyaudio.paFloat32]
    if dtype not in allowed_dtypes:
        raise TypeError("The dtype must be a one of " + str(allowed_dtypes))
    
    if np.ndim(array) != 1:
        raise TypeError("Saving multi-channel audio is not supported!")
    
    array = convert_dtype(array, dtype)
    format = allowed_formats[allowed_dtypes.index(dtype)]
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=1, rate=sample_rate, output=True)
    stream.write(array.tobytes())
    stream.stop_stream()
    stream.close()
    audio.terminate()
