import os
import librosa
import pyaudio
import pydub
import numpy as np


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


# utils

floating_dtypes = [np.float16, np.float32, np.float64]

integer_dtypes = [np.int8, np.int16, np.int32, np.int64]


def convert_dtype(array, dtype):
    dtype = np.dtype(dtype)
    
    if array.dtype == dtype:
        return array
    
    if dtype in integer_dtypes:
        item_size = dtype.itemsize
        if array.dtype in integer_dtypes:  # int -> int
            array_item_size = array.dtype.itemsize
            if array_item_size > item_size:
                array /= int(2 ** ((array_item_size - item_size) * 8))
                array = np.asarray(array, dtype)
            elif array_item_size < item_size:
                array = np.asarray(array, dtype)
                array *= int(2 ** ((item_size - array_item_size) * 8))
        elif array.dtype in floating_dtypes:  # float -> int
            array *= 2 ** (item_size * 8 - 1)
            array = np.asarray(array, dtype)
        else:
            raise TypeError("Unsupported array with dtype " + array.dtype.name)
    elif dtype in floating_dtypes:
        if array.dtype in integer_dtypes:  # int -> float
            array_item_size = array.dtype.itemsize
            array /= int(2 ** (array_item_size * 8))
            array = np.asarray(array, dtype)
        elif array.dtype in floating_dtypes:  # float -> float
            array = np.asarray(array, dtype)
        else:
            raise TypeError("Unsupported array with dtype " + array.dtype.name)
    else:
        raise TypeError("Unsupported dtype " + dtype.name)
    
    return array
