import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from data_util import *
from stock_agent import *
import tensorflow as tf
from sklearn.metrics import *

def get_max_profit(x, y, closed_prices, step):

    back = BackTest(value = 100, 
                    verbose = False, 
                    sell_on_profit = True,
                    pending_sell_steps = step)
    

    return backtest_baseline(x, y, closed_prices, step, back)

def run_trial(model, provider, step = 1):
       
    result_profits = -1
    
    reference_profit = {}
    models_profit = {}

    models_profit_metric = {}

    models_score = {}
    
    model_result = {}
    
    train_by_step(model, step, provider)
    profits = []
    for train_set in provider.val_keys:
        trainX_raw, trainY_raw = provider.load_val_data(train_set)
        x, y, closed_prices = get_sequencial_data(trainX_raw, trainY_raw, step)
        reference = get_max_profit(x, y, closed_prices, step)

        key = train_set

        reference_profit[key] = reference.get_profit()
        #print(reference.current)

        back, score = eval_step(model, train_set, step, provider)
        
        #models_profit[key] = f"{back.get_profit()}"
        models_profit[key] = back.get_profit()
        models_score[key] = score
        models_profit_metric[key] = back.get_profit() / reference.get_profit()

        profits.append(back.current)

    model_result['profit'] = np.average(profits)
    model_result['model'] = model
    model_result['models_profit'] = models_profit
    model_result['models_score'] = models_score
    model_result['models_profit_metric'] = models_profit_metric
       
    return model_result




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
        combined_data[f'direction{key}'] = np.where(combined_data[key] < 1, 1, 0)
    
    combined_data[f'direction'] = combined_data[f'direction{keys[0]}']
    for idx in range(1, len(keys)):
        combined_data[f'direction'] = combined_data[f'direction{keys[idx]}'] + combined_data[f'direction'] 
    
    combined_data[f'y'] = np.where(combined_data['direction'] > 0, 1, 0)
    
    return combined_data[f'y'].to_numpy()




def prepare_train_data(trainX, trainY, step):
    y = get_y_data(
        pd.DataFrame(trainY, columns = ['Close']), 
        (-1 * step)
    )
    x, y, closed_prices = trainX, y, trainY
    return shuffle(x, y, closed_prices, random_state = 10)


def get_sequencial_data(trainX, trainY, step):
    y = get_y_data(
        pd.DataFrame(trainY, columns = ['Close']), 
        (-1 * step)
    )
    x, y, closed_prices = trainX, y, trainY
    return x, y, closed_prices


def backtest_model(model, x, closed_prices, back):
    
    for idx in range(len(x)):
        xx = [x[idx]]
        yy = model.predict(xx)[0]
        price = closed_prices[idx]
        #print(f'{idx} {yy} {price}')
        back.on_state(0, price)
        if(yy == 1):
            back.request_buy(price)
        else:
            back.request_sell(price)
    
    #print(f"Closing backtest_model {back.current}")
    back.request_sell(price)
    #print(f"Closed backtest_model {back.current}")
    
    return back


def backtest_baseline(x, y, closed_prices, step, back):
    
    for idx in range(len(x)):
        yy = y[idx]
        price = closed_prices[idx]
        #print(f'{idx} {yy} {price}')
        back.on_state(0, price)
        #print(yy)
        if(yy == 1):
            back.request_buy(price)
        else:
            back.request_sell(price)
            
    back.request_sell(price)
    
    return back



    
def train_by_step(model, step, provider):
    trainX_raw, trainY_raw = provider.load_train_data()
    x, y, closed_prices = prepare_train_data(trainX_raw, trainY_raw, step)
    #print(f"{trainX_raw.shape}")
    #print(closed_prices)
    model.fit(x, y)
    return x, y, closed_prices

def eval_step(model, train_set, step, provider):

    valX, valY = provider.load_val_data(train_set)
    
    x, y, closed_prices = get_sequencial_data(valX, valY, step)
    
    preds = model.predict(x)

    recall = recall_score(y, preds)
    precision = precision_score(y, preds)
    f1 = f1_score(y, preds)
    accuracy = accuracy_score(y, preds)
    roc_auc = roc_auc_score(y, preds)
    
    back = BackTest(value = 100, 
                    verbose = False, 
                    sell_on_profit = True,
                    pending_sell_steps = step)
    
    back = backtest_model(model, x, closed_prices, back)
    
    return back, (precision, recall, f1, accuracy)



class MockModel():
    
    def __init__(self, 
                 model
                ):
        self.model = model
        self.normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
        
        
    def fit(self, x, y):
        self.normalizer.adapt(x)
        norm_x = self.normalizer(x).numpy()
        self.model.fit(norm_x, y)
        
    def predict(self, x):
        norm_x = self.normalizer(x).numpy()
        return self.model.predict(norm_x)
    
    def __str__(self):
        return "MockModel ( %s )" % (self.model)
    
    def __getitem__(self, i):
        return f"Value {i}"
        


class CoPilotModel():
    
    def __init__(self, 
                 model
                ):
        self.model = model
        
        
    def fit(self, x, y):
        self.model.fit(norm_x, y)
        
    def predict(self, x):
        return self.model.predict(x)
    
    def __str__(self):
        return "MockCoPilotModel ( %s )" % (self.model)
    
    def __getitem__(self, i):
        return f"Value {i}"
        
        
class MockCoPilotModel():
    
    def __init__(self, 
                 model,
                 coModel
                ):
        self.model = model
        self.coModel = coModel
        
    def compare(self, val, val2):
        return np.where(np.array(val) == np.array(val2), 1, 0)
        
    def fit(self, x, y):
        self.model.fit(x, y)
        
        preds = self.model.predict(x)
        
        coX = []
        coY = []
        yy = []
        
        for idx in range(len(x)):
            if (preds[idx] == 1):
                coX.append(x[idx])
                coY.append(preds[idx])
                yy.append(y[idx])
        
        finalCoY = self.compare(yy, coY)
        
        self.coModel.fit(coX, finalCoY)
        
    def predict(self, x):
        y = self.model.predict(x)
        
        if (y[0] == 1):
            return self.compare(y, self.coModel.predict(x))
        
        return self.model.predict(x)
    
    def __str__(self):
        return "MockCoPilotModel ( %s )" % (self.model)
        
    def __getitem__(self, i):
        return f"Value {i}"


    
class LocalDataProvider():
    
    def __init__(self, 
                train_keys,
                 val_keys
                ):
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_data = []
        self.vals = {}
    
    def load_train_data(self):
        return load_data("simple_full_", "train", path)
    
    def load_val_data(self, val):
        return load_data(f"simple_{val}", "train", path)

    
class OnLineDataProvider():
    #https://www.unixtimestamp.com/
    def __init__(self,
                 sourceDataGenerator,
                 minutes,
                 train_keys = ["btcusd", "ethusd"],
                 train_limit = 100,
                 val_limit = 1000,
                 val_keys = ["btcusd", "ethusd"],
                 val_start = 1623435037,
                 val_end = -1,
                 train_start_list = [1569969462]
                ):
        #self.train_keys = ["ltcbtc", "btceur", "btcusd", "bchusd", "ethusd", "xrpusd"]
        self.train_keys = train_keys
        #self.train_keys = ["ltcbtc", "btceur", "linkusd", "xrpusd"]
        self.val_keys = val_keys
        self.train_data = []
        self.vals = {}
        self.minutes = minutes
        self.train_limit = train_limit
        
        self.val_limit = val_limit
        self.val_start = val_start
        self.val_end = val_end
        
        self.train_start_list = train_start_list
        self.sourceDataGenerator = sourceDataGenerator
        self.steps = (minutes * 60)
        self.resample = f'{minutes}Min'
    
    def load_cache(self):
        self.load_train_cache()
        self.load_val_cache(self.val_keys, self.val_start, self.val_end)

    def load_train_cache(self):
        sets = []

        def load_from_time(time): 
            for curr in self.train_keys:
                x, closed_prices = self.sourceDataGenerator.get_full_database_online(curr, 
                                                                                resample = self.resample, 
                                                                                limit = self.train_limit,
                                                                                step = self.steps,
                                                                                start=time)
                sets.append((x, closed_prices))

        for start_time in self.train_start_list:
            load_from_time(start_time)

        self.train_data = self.sourceDataGenerator.conc_simple_sets(sets)
            
    def load_val_cache(self, val_keys, start, end):
        for key in val_keys:
            x, closed_prices = self.sourceDataGenerator.get_full_database_online(key, 
                                                                            resample = self.resample, 
                                                                            limit = self.val_limit,
                                                                            step = self.steps,
                                                                            start=start,
                                                                            end=end
                                                                           )
            self.vals[key] = (x, closed_prices)
            
    def load_train_data(self):
        return self.train_data
    
    def load_val_data(self, val):
        return self.vals[val]
    
    def report(self):
        print(f"Total train set {len(self.train_data[0])}")
        for key in self.val_keys:
            print(f"Total val {key} set {len(self.vals[key][0])}")
