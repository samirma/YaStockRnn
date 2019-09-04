from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ["TF_KERAS"] = "1"

from keras_radam import RAdam

from data_generator import DataGenerator

import data_util

import model as model_util

import matplotlib.pyplot as plt

from numpy import array
import numpy as np


import datetime
from datetime import datetime

from tensorflow.keras.layers import GaussianNoise, GlobalMaxPool1D, Bidirectional, Dense, Flatten, Conv2D, LeakyReLU, Dropout, LSTM, GRU, Input
from tensorflow.keras import Model, Sequential
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


from tensorflow.keras import datasets, layers, models

import tensorflow as tf

#%load_ext tensorboard
print(tf.__version__)




data_gen = DataGenerator(random=False, first_index=10)
data_gen.rewind()
data_gen.steps


use_cache = False #@param {type:"boolean"}
from tqdm import tqdm_notebook as tqdm

data_gen.rewind()

path = "drive/My Drive/model/"

full = data_gen.steps - 2000

trainX, trainY, valX, valY = data_util.get_sets(data_gen, full, val_percentage = 0.01, use_cache = use_cache)

