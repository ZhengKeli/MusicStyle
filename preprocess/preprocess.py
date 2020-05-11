import abc

import numpy as np


class Preprocess(abc.ABC):
    @abc.abstractmethod
    def __call__(self, wave: np.ndarray):
        pass


class ConstantQualityTransform(Preprocess):
    def __call__(self, wave: np.ndarray):
        """ convert the wave into an image
        
        :param wave: shape=[sample_count]
        :return: the spectrogram. [spectrogram_width, spectrogram_height]
        """
        # todo implement this function
        pass
