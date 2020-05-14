import os
import tensorflow as tf


def conv2d_bn(x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, name=conv_name)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization(name=bn_name)(x)
    return x


def identity_block(x, nb_filter, kernel_size, strides=(1, 1), conv_shortcut=False):
    shortcut = x
    x = conv2d_bn(x, filters=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = conv2d_bn(x, filters=nb_filter, kernel_size=kernel_size, padding='same')
    if conv_shortcut:
        shortcut = conv2d_bn(shortcut, filters=nb_filter, strides=strides, kernel_size=kernel_size)
    x = x + shortcut
    return x


def bottleneck_block(x, nb_filters, strides=(1, 1), conv_shortcut=False):
    k1, k2, k3 = nb_filters
    shortcut = x
    x = conv2d_bn(x, filters=k1, kernel_size=1, strides=strides, padding='same')
    x = conv2d_bn(x, filters=k2, kernel_size=3, padding='same')
    x = conv2d_bn(x, filters=k3, kernel_size=1, padding='same')
    if conv_shortcut:
        shortcut = conv2d_bn(shortcut, filters=k3, strides=strides, kernel_size=1)
    x = x + shortcut
    return x


def Resnet34(input_shape, classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(inputs)
    
    # conv1
    x = conv2d_bn(x, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # conv2_x
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3))
    
    # conv3_x
    x = identity_block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    x = identity_block(x, nb_filter=128, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=128, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=128, kernel_size=(3, 3))
    
    # conv4_x
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3))
    
    # conv5_x
    x = identity_block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    x = identity_block(x, nb_filter=512, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=512, kernel_size=(3, 3))
    x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(classes)(x)
    x = tf.keras.layers.Softmax()(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)


def Resnet50(input_shape, classes):
    inpt = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(inpt)
    x = conv2d_bn(x, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # conv2_x
    x = bottleneck_block(x, nb_filters=[64, 64, 256], strides=(1, 1), conv_shortcut=True)
    x = bottleneck_block(x, nb_filters=[64, 64, 256])
    x = bottleneck_block(x, nb_filters=[64, 64, 256])
    
    # conv3_x
    x = bottleneck_block(x, nb_filters=[128, 128, 512], strides=(2, 2), conv_shortcut=True)
    x = bottleneck_block(x, nb_filters=[128, 128, 512])
    x = bottleneck_block(x, nb_filters=[128, 128, 512])
    x = bottleneck_block(x, nb_filters=[128, 128, 512])
    
    # conv4_x
    x = bottleneck_block(x, nb_filters=[256, 256, 1024], strides=(2, 2), conv_shortcut=True)
    x = bottleneck_block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_block(x, nb_filters=[256, 256, 1024])
    
    # conv5_x
    x = bottleneck_block(x, nb_filters=[512, 512, 2048], strides=(2, 2), conv_shortcut=True)
    x = bottleneck_block(x, nb_filters=[512, 512, 2048])
    x = bottleneck_block(x, nb_filters=[512, 512, 2048])
    
    x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inpt, outputs=x)


def Resnet34_modified(input_shape, classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    
    # conv1
    x = conv2d_bn(x, filters=64, kernel_size=(5, 11), strides=(1, 2), padding='same')
    
    # conv2_x
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3))
    
    # conv3_x
    x = identity_block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 2), conv_shortcut=True)
    x = identity_block(x, nb_filter=128, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=128, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=128, kernel_size=(3, 3))
    
    # conv4_x
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3))
    
    # conv5_x
    x = identity_block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    x = identity_block(x, nb_filter=512, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=512, kernel_size=(3, 3))
    
    # tail
    x = tf.keras.layers.AveragePooling2D(pool_size=(3, 7))(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(classes)(x)
    x = tf.keras.layers.Softmax()(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)
