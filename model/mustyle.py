import tensorflow as tf


class MuStyleModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def call(self, inputs, training=None, mask=None):
        # todo
        # inputs [batch_size, sample_count, frequency_count]
        # outputs [batch_size, type_count]
        
        img = inputs
        return super().call(inputs, training, mask)
