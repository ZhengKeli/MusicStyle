import tensorflow as tf


# v1

def sp_block_v1(sp, regularizer=None):
    x = sp
    
    x = tf.keras.layers.Conv2D(
        filters=16, kernel_size=3, padding='same',
        kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same',
                               kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                               kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same',
                               kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same',
                               kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.transpose(x, [0, 2, 1, 3])
    x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], 2 * 16]))(x)
    
    return x


def SpNet_v1(sp1_shape, sp2_shape, classes):
    regularizer = tf.keras.regularizers.l2(0.01)
    
    sp1_input = tf.keras.layers.Input(shape=sp1_shape)
    sp1 = sp1_input
    sp1 = tf.keras.layers.GaussianNoise(0.03)(sp1)
    sp1 = sp_block_v1(sp1, regularizer)
    
    sp2_input = tf.keras.layers.Input(shape=sp2_shape)
    sp2 = sp2_input
    sp1 = tf.keras.layers.GaussianNoise(0.03)(sp1)
    sp2 = sp_block_v1(sp2, regularizer)
    
    x = tf.concat([sp1, sp2], -1)
    x = tf.keras.layers.Conv1D(64, 7, padding='valid')(x)
    x = tf.keras.layers.ReLU(classes)(x)
    x = tf.keras.layers.MaxPool1D(7, strides=2)(x)
    x = tf.reduce_mean(x, axis=-2)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(classes)(x)
    x = tf.keras.layers.Softmax()(x)
    
    return tf.keras.Model(inputs=[sp1_input, sp2_input], outputs=x)


# v2

def sp_block_v2(sp, regularizer=None):
    x = sp
    
    x = tf.keras.layers.Conv2D(
        filters=16, kernel_size=3, padding='same',
        kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same',
                               kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                               kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same',
                               kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.transpose(x, [0, 2, 1, 3])
    x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], 3 * 32]))(x)
    
    return x


def SpNet_v2(sp1_shape, sp2_shape, classes):
    regularizer = tf.keras.regularizers.l2(0.02)
    
    sp1_input = tf.keras.layers.Input(shape=sp1_shape)
    sp1 = sp1_input
    sp1 = tf.keras.layers.GaussianNoise(0.1)(sp1)
    sp1 = sp_block_v2(sp1, regularizer)
    
    sp2_input = tf.keras.layers.Input(shape=sp2_shape)
    sp2 = sp2_input
    sp1 = tf.keras.layers.GaussianNoise(0.1)(sp1)
    sp2 = sp_block_v2(sp2, regularizer)
    
    x = tf.concat([sp1, sp2], -1)
    x = tf.keras.layers.Conv1D(64, 7, padding='valid', kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.ReLU(classes)(x)
    x = tf.keras.layers.MaxPool1D(7, strides=2)(x)
    x = tf.reduce_mean(x, axis=-2)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(classes, kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Softmax()(x)
    
    return tf.keras.Model(inputs=[sp1_input, sp2_input], outputs=x)
