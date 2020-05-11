import tensorflow as tf


class Classifier(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        """
        :param inputs: shape=[batch_size, vector_size]
        :return: onehot. shape=[batch_size, type_count]
        """
        # todo implement this method
        return super().call(inputs, **kwargs)
