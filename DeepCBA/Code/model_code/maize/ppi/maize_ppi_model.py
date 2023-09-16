# -*- coding: utf-8 -*-
'''
@ author LiJ
@ version 1.0
@ date 2023 / 09 / 14 15: 11
@ Description: maize_ppi_model
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, RepeatVector


def split(tensor):
    import tensorflow as tf
    ret = tf.split(tensor, axis=2, num_or_size_splits=2)
    return ret


def build_1(l):
    l = keras.layers.Conv2D(64, kernel_size=(4, 8), padding='valid', input_shape=[4, 3000, 1])(l)
    l = keras.layers.BatchNormalization(momentum=0.99, scale=False)(l)
    l = keras.layers.ReLU()(l)
    l = keras.layers.Conv2D(64, kernel_size=(1, 8), padding='same')(l)
    l = keras.layers.BatchNormalization(momentum=0.99, scale=False)(l)
    l = keras.layers.ReLU()(l)
    l = keras.layers.MaxPooling2D(pool_size=(1, 8), strides=(1, 8), padding='same')(l)
    l = keras.layers.Dropout(0.3)(l)

    l = keras.layers.Conv2D(128, kernel_size=(1, 8), padding='same')(l)
    l = keras.layers.BatchNormalization(momentum=0.99, scale=False)(l)
    l = keras.layers.ReLU()(l)
    l = keras.layers.Conv2D(128, kernel_size=(1, 8), padding='same')(l)
    l = keras.layers.BatchNormalization(momentum=0.99, scale=False)(l)
    l = keras.layers.ReLU()(l)
    l = keras.layers.MaxPooling2D(pool_size=(1, 8), strides=(1, 8), padding='same')(l)
    l = keras.layers.Dropout(0.3)(l)

    l = keras.layers.Conv2D(64, kernel_size=(1, 8), padding='same')(l)
    l = keras.layers.BatchNormalization(momentum=0.99, scale=False)(l)
    l = keras.layers.ReLU()(l)
    l = keras.layers.Conv2D(64, kernel_size=(1, 8), padding='same')(l)
    l = keras.layers.BatchNormalization(momentum=0.99, scale=False)(l)
    l = keras.layers.ReLU()(l)
    l = keras.layers.MaxPooling2D(pool_size=(1, 8), strides=(1, 8), padding='same')(l)
    l = keras.layers.Dropout(0.3)(l)

    l = keras.layers.Reshape((3, 128))(l)
    l = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(l)
    return l


def buildModel():
    model_input = tf.keras.Input(shape=(4, 6000, 1))
    model_a, model_b = tf.keras.layers.Lambda(split)(model_input)
    modela = build_1(model_a)
    modelb = build_1(model_b)

    l = tf.keras.layers.Attention()([modela, modelb])
    l = keras.layers.Flatten()(l)
    l = keras.layers.Dropout(0.3)(l)
    l = keras.layers.Dense(64, activation='relu')(l)
    l = keras.layers.Dense(1, activation='linear')(l)

    model = keras.Model(inputs=model_input,outputs=l)
    return model

# model=buildModel()
# model.summary()