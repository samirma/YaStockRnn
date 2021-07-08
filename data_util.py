from tqdm.notebook import tqdm
import os
import state_util
import numpy as np
from state_util import StateUtil
import random

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator, pad_sequences

from collections import Counter
from imblearn.over_sampling import RandomOverSampler

on_state_parsed = lambda list, price, amount: list

def get_set(set_name, data_count, data_gen, path = "drive/My Drive/model/", use_cache=True, on_state_parsed = on_state_parsed):
    trainX = []
    trainY = []
    x_path = path + set_name + "X.npy"
    y_path = path + set_name + "Y.npy"
    
    stateUtil = StateUtil(data_gen = data_gen, on_state_parsed = on_state_parsed)
    
    if (use_cache and os.path.exists(x_path) and (os.path.exists(y_path))):
        print("Loading data from files {} {}".format(x_path, y_path))
        trainX = np.load(x_path)
        trainY = np.load(y_path)
    else:
        old_state = []
        for i in tqdm(range(data_count)):
            raw = data_gen.next()
            try:
                state = stateUtil.get_state(raw, data_gen.index)
            except:
                #print("#same state")
                continue
            
            if (old_state == state[0]):
                #print("same state, ignored")
                continue
            old_state = state[0]
            trainX.append(state[0])
            trainY.append(state[1])
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    np.save(x_path, trainX)
    np.save(y_path, trainY)

    return trainX, trainY

def get_sets(data_gen, data_count, val_percentage = 0.03, path = "drive/My Drive/model/", use_cache=True, on_state_parsed = on_state_parsed):
    trainX, trainY = get_set("train", int(data_count*(1-val_percentage)), data_gen,  path, use_cache, on_state_parsed)
    valX, valY = get_set("val", int(data_count*val_percentage), data_gen,  path, use_cache, on_state_parsed)
    print("Completed: {} {} {} {}".format(trainX.shape, trainY.shape, valX.shape, valY.shape))
    return trainX, trainY, valX, valY


def load_raw_data(name, sufix, path):
    X = np.load(path + name + sufix + "X.npy", allow_pickle=True)
    Y = np.load(path + name + sufix + "Y.npy", allow_pickle=True)
    return X, Y

def get_balanced_set(x, y):
    oversample = RandomOverSampler(sampling_strategy='minority')
    #print(Counter(y))
    X_over, y_over = oversample.fit_resample(x, y)
    #print(Counter(y_over))
    return X_over, y_over

def get_balanced_set_seq(x, y):
    #oversample = RandomOverSampler(sampling_strategy='minority')
    #print(Counter(y))
    #X_over, y_over = oversample.fit_resample(x, y)
    #print(Counter(y_over))
    train_features = x
    train_labels = y
    
    bool_train_labels = train_labels != 0

    pos_features = train_features[bool_train_labels]
    neg_features = train_features[~bool_train_labels]

    pos_labels = train_labels[bool_train_labels]
    neg_labels = train_labels[~bool_train_labels]

    ids = np.arange(len(pos_features))
    choices = np.random.choice(ids, len(neg_features))

    res_pos_features = pos_features[choices]
    res_pos_labels = pos_labels[choices]

    res_pos_features.shape

    resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
    resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

    order = np.arange(len(resampled_labels))
    np.random.shuffle(order)
    resampled_features = resampled_features[order]
    resampled_labels = resampled_labels[order]

    return resampled_features, resampled_labels


def load_data(name, sufix, path = "data/"):
    x, y = load_raw_data(name, sufix, path)

    return x, y


def prepare_y(list, null_value):
    empty = np.array([null_value])
    list = np.concatenate((empty, list), axis=0)
    list = np.delete(list , -1, 0)
    return list

def create_dataset(X, y, time_steps=1, null_value = [0, 0]):
    y = prepare_y(y, null_value)
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def get_gen(set_x, set_y, shuffle=True, batch_size=64, time_steps=30):
    set_y = prepare_y(set_y)
    return TimeseriesGenerator(set_x, set_y, length=time_steps, batch_size=batch_size, shuffle=shuffle)
