import os
import tensorflow as tf


def conv2d_bn(x, filters, kernel_size, strides=(1, 1), padding='same', regularizer=None, name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides,
                               kernel_regularizer=regularizer,
                               name=conv_name)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization(name=bn_name)(x)
    return x


def identity_block(x, nb_filter, kernel_size, strides=(1, 1), regularizer=None, conv_shortcut=False):
    shortcut = x
    x = conv2d_bn(x, filters=nb_filter, kernel_size=kernel_size, strides=strides,
                  padding='same', regularizer=regularizer)
    x = conv2d_bn(x, filters=nb_filter, kernel_size=kernel_size,
                  padding='same', regularizer=regularizer)
    if conv_shortcut:
        shortcut = conv2d_bn(shortcut, filters=nb_filter, strides=strides, kernel_size=kernel_size,
                             padding='same', regularizer=regularizer)
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


def Resnet011(input_shape, classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    
    x = conv2d_bn(x, filters=32, kernel_size=(5, 9), strides=(1, 2), padding='same')
    x = identity_block(x, nb_filter=32, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=32, kernel_size=(3, 3))
    
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3), strides=(1, 2), conv_shortcut=True)
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3))
    
    x = identity_block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    x = identity_block(x, nb_filter=128, kernel_size=(3, 3))
    
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3))
    
    # tail
    x = tf.keras.layers.AveragePooling2D(pool_size=(3, 7))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(classes)(x)
    x = tf.keras.layers.Softmax()(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)


# train99 test68 val62
def Resnet020(sp_shape, classes):
    sp = tf.keras.layers.Input(shape=sp_shape)
    x = sp
    
    x = conv2d_bn(x, filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same')
    x = identity_block(x, nb_filter=32, kernel_size=(3, 3))
    
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3))
    
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3))
    
    x = tf.reduce_mean(x, axis=-2)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(classes)(x)
    x = tf.keras.layers.Softmax()(x)
    
    return tf.keras.Model(inputs=sp, outputs=x)


# train91 test67 val62
def Resnet021(sp_shape, classes):
    sp = tf.keras.layers.Input(shape=sp_shape)
    x = sp
    
    x = conv2d_bn(x, filters=32, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    
    x = tf.transpose(x, [0, 2, 1, 3])
    x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], 10 * 64]))(x)
    x = tf.keras.layers.MaxPool1D(7, 2)(x)
    x = tf.reduce_mean(x, axis=-2)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes)(x)
    x = tf.keras.layers.Softmax()(x)
    
    return tf.keras.Model(inputs=sp, outputs=x)


# train99 test67 val62 (cqt) 63 (mfcc)
def Resnet022(sp_shape, classes):
    sp = tf.keras.layers.Input(shape=sp_shape)
    x = sp
    
    x = conv2d_bn(x, filters=32, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    
    x = tf.transpose(x, [0, 2, 1, 3])
    x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], 10 * 64]))(x)
    x = tf.keras.layers.MaxPool1D(7, 2)(x)
    x = tf.reduce_mean(x, axis=-2)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(classes)(x)
    x = tf.keras.layers.Softmax()(x)
    
    return tf.keras.Model(inputs=sp, outputs=x)


def Resnet030(sp_shape, fe_shape, classes):
    sp = tf.keras.layers.Input(shape=sp_shape)
    x = sp
    
    x = conv2d_bn(x, filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same')
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    
    x = tf.reduce_mean(x, axis=-2)
    x = tf.keras.layers.Flatten()(x)
    
    fe = tf.keras.layers.Input(shape=fe_shape)
    z = fe
    
    z = tf.keras.layers.Dense(256, activation='relu')(z)
    
    y = tf.concat([x, z], -1)
    y = tf.keras.layers.Dense(128, activation='relu')(y)
    y = tf.keras.layers.Dense(64, activation='relu')(y)
    y = tf.keras.layers.Dense(classes)(y)
    y = tf.keras.layers.Softmax()(y)
    
    return tf.keras.Model(inputs=[sp, fe], outputs=y)


def Resnet031(sp_shape, fe_shape, classes):
    # sp
    sp = tf.keras.layers.Input(shape=sp_shape)
    x = sp
    
    x = conv2d_bn(x, filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same')
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    
    x = tf.reduce_mean(x, axis=-2)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    
    # fe
    fe = tf.keras.layers.Input(shape=fe_shape)
    z = fe
    
    z = tf.keras.layers.Dense(256, activation='relu')(z)
    z = tf.keras.layers.Dense(128, activation='relu')(z)
    z = tf.keras.layers.Dense(32, activation='relu')(z)
    
    # soft max
    y = x + z
    y = tf.keras.layers.Dense(classes)(y)
    y = tf.keras.layers.Softmax()(y)
    
    return tf.keras.Model(inputs=[sp, fe], outputs=y)


# train99 test69 val65
def Resnet040(sp1_shape, sp2_shape, classes):
    sp1_input = tf.keras.layers.Input(shape=sp1_shape)
    sp1 = sp1_input
    sp1 = conv2d_bn(sp1, filters=32, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    sp1 = identity_block(sp1, nb_filter=32, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    sp1 = identity_block(sp1, nb_filter=32, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    sp1 = tf.transpose(sp1, [0, 2, 1, 3])
    sp1 = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], 10 * 32]))(sp1)
    
    sp2_input = tf.keras.layers.Input(shape=sp2_shape)
    sp2 = sp2_input
    sp2 = conv2d_bn(sp2, filters=32, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    sp2 = identity_block(sp2, nb_filter=32, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    sp2 = identity_block(sp2, nb_filter=32, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    sp2 = tf.transpose(sp2, [0, 2, 1, 3])
    sp2 = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], 10 * 32]))(sp2)
    
    x = tf.concat([sp1, sp2], -1)
    x = tf.keras.layers.Conv1D(64, 7, padding='valid')(x)
    x = tf.keras.layers.MaxPool1D(7, strides=2)(x)
    x = tf.reduce_mean(x, axis=-2)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(classes)(x)
    x = tf.keras.layers.Softmax()(x)
    
    return tf.keras.Model(inputs=[sp1_input, sp2_input], outputs=x)
