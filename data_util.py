from tqdm import tqdm_notebook as tqdm
import os
import state_util
import numpy as np
from state_util import StateUtil
import random
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator, pad_sequences

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
        for i in tqdm(range(data_count)):
            raw = data_gen.next()
            state = stateUtil.get_state(raw, data_gen.index)
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

def get_balanced_set(X, Y):
    positiveX = []
    positiveY = []
    negativeX = []
    negativeY = []

    for i in range(1, len(X)-1):
        x = X[i]
        y = Y[i]
        if (y[1] == 1):
            positiveX.append(x)
            positiveY.append(y)
        else:
            negativeX.append(x)
            negativeY.append(y)
   
    positiveX = np.array(positiveX)
    positiveY = np.array(positiveY)
    negativeX = np.array(negativeX)
    negativeY = np.array(negativeY)
    
    #print("Positives: {} Negatives {}".format(len(positiveX), len(negativeX)))
    
    negative_num = len(positiveY)
    start_index = random.randint(0,(len(negativeX) - negative_num))
    end_index = start_index + negative_num
    
    #print("Ramdow {} {}".format(start_index, end_index))
    
    trainX = np.concatenate((positiveX, negativeX[start_index:end_index]), axis=0)
    trainY = np.concatenate((positiveY, negativeY[start_index:end_index]), axis=0)
    return trainX, trainY, positiveX, positiveY, negativeX, negativeY


def load_data(name, sufix, path, balanced):
    x, y = load_raw_data(name, sufix, path)
    positiveX = []
    positiveY = []
    negativeX = []
    negativeY = []
    if (balanced):
        x, y, positiveX, positiveY, negativeX, negativeY = get_balanced_set(x, y)
        
    #print("Loaded: {} {} ".format(x.shape, y.shape))

    return x, y, positiveX, positiveY, negativeX, negativeY


def prepare_y(list):
    empty = np.array([[0, 0]])
    list = np.concatenate((empty, list), axis=0)
    list = np.delete(list , -1, 0)
    return list

def create_dataset(X, y, time_steps=1):
    y = prepare_y(y)
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def get_gen(set_x, set_y, shuffle=True, batch_size=64, time_steps=30):
    set_y = prepare_y(set_y)
    return TimeseriesGenerator(set_x, set_y, length=time_steps, batch_size=batch_size, shuffle=shuffle)
