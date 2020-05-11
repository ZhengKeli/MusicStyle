import tensorflow as tf


def ResNet50(input_shape):
    return tf.keras.applications.resnet.ResNet50(
        include_top=False,
        pooling='max',
        input_shape=input_shape,
        weights=None,
    )
