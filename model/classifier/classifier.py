import tensorflow as tf


class Classifier(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs, **kwargs):
        # todo
        # inputs: [batch_size, vector_size]
        # outputs: [batch_size, type_count] (onehot)
        return super().call(inputs, **kwargs)
