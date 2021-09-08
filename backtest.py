import pandas as pd
from data_util import *

def get_max_profit(x, y, closed_prices, step, stop_loss):

    back = BackTest(value = 100, 
                    verbose = False, 
                    sell_on_profit = True,
                    stop_loss=stop_loss,
                    pending_sell_steps = step)
    

    return backtest_baseline(x, y, closed_prices, step, back)


def prepare_train_data(trainX, trainY, step):
    x, y, closed_prices = get_sequencial_data(trainX, trainY, step)
    #x, y = get_balanced_set(x, y)
    return x, y
    #return shuffle(x, y, random_state = 10)

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
    
def train_by_step(model, step, provider):
    trainX_raw, trainY_raw = provider.load_train_data()
    x, y = prepare_train_data(trainX_raw, trainY_raw, step)
    model.fit(x, y)
    return x, y

