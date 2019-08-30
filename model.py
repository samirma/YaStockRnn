from numpy import array
import numpy as np


import datetime
from datetime import datetime

from tensorflow.keras.layers import GlobalMaxPool1D, Bidirectional, Dense, Flatten, Conv2D, LeakyReLU, Dropout, LSTM, GRU, Input
from tensorflow.keras import Model, Sequential

from tensorflow.keras import datasets, layers, models

import tensorflow as tf

def get_model(n_input, features):
    inputX = Input(shape=(n_input,features))

    def add_deep_layers(input_layer):
        x = Dense(300)(input_layer)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        x_shorcut = x
        x_shorcut = Dense(300)(x_shorcut)
        x_shorcut = tf.keras.layers.LeakyReLU(alpha=0.3)(x_shorcut)
        x = Dense(300)(x)
        x = Dropout(0.2)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        x = Dense(300)(x)
        x = Dropout(0.2)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        x = layers.add([x, x_shorcut])
        return x


    #x = Bidirectional(LSTM(features*20, return_sequences=True))(inputX)
    x = Bidirectional(LSTM(250, return_sequences=True))(inputX)
    x = Bidirectional(layers.LSTM(100))(x)

    #lstm = LSTM(features*40)(inputX)
    #x = add_deep_layers(lstm)

    #x = Flatten()(inputX)
    x = add_deep_layers(x)

    #x = layers.add([lstm, x])
    x = add_deep_layers(x)
    x = add_deep_layers(x)
    x = add_deep_layers(x)
    x = add_deep_layers(x)
    x = Dense(300)(x)
    x = Dropout(0.2)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = Dense(100)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=[inputX], outputs=x)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001, epsilon=1e-6),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

    
def load_model(filepath = "drive/My Drive/model/stock.h5"):
    model_loaded = tf.keras.models.load_model(filepath)
    return model_loaded
