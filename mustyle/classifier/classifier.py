import tensorflow as tf


class Classifier(tf.keras.Model):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # todo place all weights here
    
    def call(self, inputs, training=None, mask=None):
        # todo calculate outputs
        return super().call(inputs, training, mask)
