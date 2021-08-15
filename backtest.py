import pandas as pd
import numpy as np
from data_util import *
from stock_agent import *
import tensorflow as tf
from sklearn.metrics import *
from sklearn.utils import shuffle

def get_max_profit(x, y, closed_prices, step):

    back = BackTest(value = 100, 
                    verbose = False, 
                    sell_on_profit = True,
                    pending_sell_steps = step)
    

    return backtest_baseline(x, y, closed_prices, step, back)

def run_trial(model, provider, step):
           
    reference_profit = {}
    models_profit = {}

    models_profit_metric = {}

    models_score = {}
    
    model_result = {}
    
    profits = []
    for train_set in provider.val_keys:
        trainX_raw, trainY_raw, times = provider.load_val_data(train_set)
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
    model_result['reference_profit'] = reference_profit
       
    return model_result

def prepare_train_data(trainX, trainY, step):
    y = get_y_data(
        pd.DataFrame(trainY, columns = ['Close']), 
        (-1 * step)
    )
    #x, y = series_to_supervised(trainX, n_in=2), y
    x, y = get_balanced_set(trainX, y)
    return x, y
    #return shuffle(x, y, random_state = 10)


def get_sequencial_data(trainX, trainY, step):
    y = get_y_data(
        pd.DataFrame(trainY, columns = ['Close']), 
        (-1 * step)
    )
    #x, y, closed_prices = series_to_supervised(trainX, n_in=2), y, trainY
    x, y, closed_prices = trainX, y, trainY
    return x, y, closed_prices


def backtest_model(model, x, closed_prices, back: BackTest):
    limit = len(x)
    #if (limit > 3000):
    #    limit = 3000
    for idx in range(limit):
        xx = np.array([x[idx]])
        yy = model.predict(xx)[0]
        price = closed_prices[idx]
        #print(f'{idx} {yy} {price}')
        if(yy == 1):
            back.on_up(price, price)
        else:
            back.on_down(price, price)
    
    #print(f"Closing backtest_model {back.current}")
    back.sell(price)
    #print(f"Closed backtest_model {back.current}")
    
    return back


def backtest_baseline(x, y, closed_prices, step, back: BackTest):
    limit = len(x)
    if (limit > 3000):
        limit = 3000
    for idx in range(limit):
        yy = y[idx]
        price = closed_prices[idx]
        #print(f'{idx} {yy} {price}')
        #print(yy)
        if(yy == 1):
            back.on_up(price, price)
        else:
            back.on_down(price, price)
            
    back.on_down(price, price)
    
    return back

def test_model(model, set_key, provider, step, verbose = True):
    valX, valY, times = provider.load_val_data(set_key)

    x, y, closed_prices = get_sequencial_data(valX, valY, step)
    
    #print(len(x))
    #preds = model.predict(x)

    #recall = recall_score(y, preds)
    #precision = precision_score(y, preds)
    #f1 = f1_score(y, preds)
    #accuracy = accuracy_score(y, preds)

    back = BackTest(value = 100, 
                    verbose = verbose, 
                    sell_on_profit = True,
                    pending_sell_steps = step)
    back = backtest_model(model, x, closed_prices, back)
    return back
    
def train_by_step(model, step, provider):
    trainX_raw, trainY_raw = provider.load_train_data()
    x, y = prepare_train_data(trainX_raw, trainY_raw, step)
    model.fit(x, y)
    return x, y

def eval_step(model, train_set, step, provider, verbose = False):

    valX, valY, times = provider.load_val_data(train_set)
    
    x, y, closed_prices = get_sequencial_data(valX, valY, step)
    
    preds = model.predict(x)
    
    metrics = {}
    metrics["recall"] = recall_score(y, preds)
    metrics["precision"] = precision_score(y, preds)
    metrics["f1"] = f1_score(y, preds)
    metrics["accuracy"] = accuracy_score(y, preds)
    #metrics["roc_auc"] = roc_auc_score(y, preds)
    
    back = BackTest(value = 100, 
                    verbose = verbose, 
                    sell_on_profit = True,
                    pending_sell_steps = step)
    
    back = backtest_model(model, x, closed_prices, back)
    
    return back, metrics
    