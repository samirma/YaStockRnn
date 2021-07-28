from tqdm.notebook import tqdm
import os
import numpy as np
from data_generator import *
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator, pad_sequences
from bitstamp import *
from data_agent import *

from collections import Counter
from imblearn.over_sampling import RandomOverSampler

class SourceDataGenerator():
    def __init__(self,
                 tec,
                 base_dir = "data/"
                ):
        
        self.base_dir = base_dir
        self.tec = tec

            
    def save(self, data_set, prefix = ""):
        trainX = data_set[0]
        trainY = data_set[1]
        final_path = self.base_dir
        train_path = "{}{}trainX.npy".format(final_path, prefix)
        np.save(train_path, trainX)
        np.save("{}{}trainY.npy".format(final_path, prefix), trainY)
        if (len(data_set) > 2):
            valX = data_set[2]
            valY = data_set[3]
            np.save("{}{}valX.npy".format(final_path, prefix), valX)
            np.save("{}{}valY.npy".format(final_path, prefix), valY)
        print("Saving {} with {}".format(train_path, trainX.shape))

    def get_y_data_old(self, ohlc, shift = -1):
        combined_data = ohlc.copy()
        #combined_data['return'] = np.log(combined_data / combined_data.shift(1))
        returns = (ohlc / ohlc.shift(shift))
        combined_data['return'] = returns
        combined_data['direction'] = np.where(combined_data['return'] < 1, 1, 0)
        #print(combined_data)
        #combined_data.dropna(inplace=True)
        #print(combined_data[20:40])
        #
        return combined_data['direction'].to_numpy()

    def split(self, x, y, split, shuffle=False):
        trainX, valX, trainY, valY = train_test_split(np.array(x), np.array(y), test_size=split, shuffle=shuffle)
        print("Completed: {} {} {} {}".format(trainX.shape, trainY.shape, valX.shape, valY.shape))
        return trainX, trainY, valX, valY


    def get_full_database(self, resample, raw_dir):

        full_data = self.base_dir + raw_dir + "/"
        data_gen = DataGenerator(random = False, base_dir = full_data)
        data_gen.rewind()
        data_count = (data_gen.steps - 100)
        #data_count = 200000

        final_x = []

        closed_prices = []

        on_new_data = lambda x: final_x.append(x)
        on_closed_price = lambda price: closed_prices.append(price)

        agent = DataAgent(
            tec = self.tec,
            resample = resample,
            on_new_data = on_new_data,
            on_closed_price = on_closed_price
        )

        print("Processing {}".format(raw_dir))

        for i in tqdm(range(data_count)):
            agent.on_new_raw_data(data_gen.next())


        closes = pd.DataFrame(closed_prices, columns = ['Close'])

        final_y = get_y_data(closes, -1)

        #print(agent.ohlc)

        return final_x, final_y, closed_prices


    def load_dataset(self, dir):
        load_datasets([dir])

    def load_datasets(self, dirs, resample):
        print(dirs)
        sets = []  
        for raw_dir in dirs:

            x, y, closed_prices = self.get_full_database(resample = resample,
                                                 raw_dir = raw_dir)

            final_data = split(x, y, 0.1, shuffle=False)

            save(final_data, raw_dir)
            sets.append((x, y))
        return sets

    def conc_sets(self, sets):
        trainX = sets[0][0]
        trainY = sets[0][1]
        for i in range(1,  len(sets)):
            data_set = sets[i]
            trainX = np.append(data_set[0], trainX, axis = 0)
            trainY = np.append(data_set[1], trainY, axis = 0)

        trainX, trainY, valX, valY = split(trainX, trainY, 0.1, shuffle=True)

        return trainX, trainY, valX, valY


    def load_simple_datasets(self, dirs, resample):
        print(f"Simple: {dirs}")
        sets = []  
        for raw_dir in dirs:

            x, y, closed_prices = self.get_full_database(resample = resample,
                                                 raw_dir = raw_dir)

            save((np.array(x), closed_prices), f"simple_{raw_dir}")
            sets.append((x, closed_prices))
        return sets

    def conc_simple_sets(self, sets):
        trainX = sets[0][0]
        trainY = sets[0][1]
        for i in range(1,  len(sets)):
            data_set = sets[i]
            trainX = np.append(data_set[0], trainX, axis = 0)
            trainY = np.append(data_set[1], trainY, axis = 0)

        return trainX, trainY
    
    
    def parse(self, parsed):
        for i in parsed: #observe all columns
            timestamp = datetime.datetime.fromtimestamp(int(i['timestamp']))
            #print(timestamp,i['high'],i['low'],i['open'],i['close'],i['volume'])

        # fill in a DF with the extracted data
        df = pd.DataFrame(parsed)
        return df

    def generate_simple_data(self, parsed):
        df = pd.DataFrame(parsed).copy()
        CLOSE = 'close'
        OPEN = 'open'
        df[CLOSE] = df[CLOSE].astype(float)
        df[OPEN] = df[OPEN].astype(float)
        return df[CLOSE][:-1], df.shift(-1)[OPEN][:-1]


    def process_online_data(self, result, resample, currency):
        init = datetime.datetime.fromtimestamp(int(result[0]['timestamp']))
        end = datetime.datetime.fromtimestamp(int(result[-1]['timestamp']))
        print(f"Downloaded from {init} to {end} {result[-1]['open']}")

        value, open_value  = self.generate_simple_data(result)

        final_x = []

        closed_prices = []

        on_new_data = lambda x: final_x.append(x)
        on_closed_price = lambda price: price

        agent = DataAgent(
            tec = self.tec,
            resample = resample,
            on_new_data = on_new_data,
            on_closed_price = on_closed_price
        )

        print("Processing {} of {}".format(len(value), currency))

        for idx in tqdm(range(len(value))):
            x = float(value[idx])
            closed_prices.append(open_value[idx])
            agent.on_new_price(x)

        closes = pd.DataFrame(closed_prices, columns = ['Close'])

        #print(agent.ohlc)

        return np.array(final_x), np.array(closed_prices)

    def get_full_database_online(self, currency, resample, start=1619823600, end=-1, step=60, limit=10):

        result = load_bitstamp_ohlc(currency, 
                                    start=start,
                                    end=end,
                                    step=step, 
                                    limit=limit)

        return self.process_online_data(result, resample, currency)

    def get_full_database_online_period(self, currency, resample, start, end, step=60, limit=1000):
    
        result = load_bitstamp_ohlc_by_period(currency, 
                                    start=start,
                                    end=end,
                                    step=step)

        return self.process_online_data(result, resample, currency)






def get_online_data(minutes, source_data_generator, load_from_disk, file_prefix = ""):
    
    online = OnLineDataProvider(
                 sourceDataGenerator = source_data_generator,
                 minutes = minutes,
                 train_keys = train_keys,
                 train_limit = 1000,
                 val_limit = 1000,
                 val_keys = ["btcusd"],
                 val_start = val_start,
                 val_end = val_end,
                 train_start_list = train_start_list
    )

    online_path = f'data/online{file_prefix}_{minutes}'
    
    if (load_from_disk):
        online = load(online_path)    
    else:
        #online.load_train_cache()
        online.load_cache()
        online.sourceDataGenerator = None
        dump(online, online_path)
        
    
    return online



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

def get_balanced_set(x, y, sampling_strategy='minority'):
    oversample = RandomOverSampler(sampling_strategy=sampling_strategy)
    print(Counter(y))
    X_over, y_over = oversample.fit_resample(x, y)
    print(Counter(y_over))
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


def get_y_data(ohlc, shift = -1):
    combined_data = ohlc.copy()
    #combined_data['return'] = np.log(combined_data / combined_data.shift(1))
    
    keys = []
    steps = (shift * -1) + 1
    for idx in range(1, steps):
        returns = (ohlc / ohlc.shift(-1 * idx))
        key = f'{idx}'
        keys.append(key)
        combined_data[key] = returns
    
    for key in keys:
        #combined_data[f'direction{key}'] = np.where(combined_data[key] < 1, 1, 0)
        combined_data[f'direction{key}'] = np.where(combined_data[key] < 0.999, 1, 0)

    
    combined_data[f'direction'] = combined_data[f'direction{keys[0]}']
    for idx in range(1, len(keys)):
        combined_data[f'direction'] = combined_data[f'direction{keys[idx]}'] + combined_data[f'direction'] 
    
    combined_data[f'y'] = np.where(combined_data['direction'] > 0, 1, 0)
    
    return combined_data[f'y'].to_numpy()
