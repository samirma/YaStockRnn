from tqdm import tqdm
import os
import state_util
import numpy as np
from state_util import StateUtil

def get_set(set_name, data_count, data_gen, path = "drive/My Drive/model/"):
    trainX = []
    trainY = []
    x_path = path + set_name + "X.npy"
    y_path = path + set_name + "Y.npy"
    
    stateUtil = StateUtil()
    
    if (os.path.exists(x_path) and (os.path.exists(y_path))):
        print("Loading data from files {} {}".format(x_path, y_path))
        trainX = np.load(x_path)
        trainY = np.load(y_path)
    else:
        for i in tqdm(range(data_count)):
            state = data_gen.next()
            state = stateUtil.get_state(state, data_gen)
            trainX.append(state[0])
            trainY.append(state[1])
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    np.save(x_path, trainX)
    np.save(y_path, trainY)

    return trainX, trainY

def get_sets(data_gen, data_count, val_percentage = 0.03, path = "drive/My Drive/model/"):
    trainX, trainY = get_set("train", int(data_count*(1-val_percentage)), data_gen,  path)
    valX, valY = get_set("val", int(data_count*val_percentage), data_gen,  path)
    print(trainX.shape)
    print(trainY.shape)
    print(valX.shape)
    print(valY.shape)
    return trainX, trainY, valX, valY




