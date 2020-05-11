import tensorflow as tf


class ImgProcess(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        """
        :param inputs: [batch_size, spectrogram_width, spectrogram_height]
        :return: [batch_size, vector_size]
        """
        # todo implement this method
        return super().call(inputs, **kwargs)


def ResNet50(input_shape):
    return tf.keras.applications.resnet.ResNet50(
        include_top=False,
        pooling='max',
        input_shape=input_shape,
        weights=None,
    )
