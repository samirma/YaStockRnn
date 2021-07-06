import pandas as pd
import data_util
from tqdm import tqdm_notebook as tqdm
from data_generator import DataGenerator
from tec_an import TecAn
from data_agent import DataAgent, TacProcess
import numpy as np
from data_util import *
from sklearn_model_hyper import *
import pandas as pd


class BackTest():
    
    def __init__(self, 
                 initial_value = 100, 
                 value = 100,
                 verbose = False
                ):
        self.initial_value = initial_value
        self.value = value
        self.verbose = verbose
        self.reset()
        
        
    def reset(self):
        self.current = self.initial_value
        self.holding = 0
        self.bought = False
        self.buy_price = 0
        self.timestamp = 0
        
    def request_buy(self, ask):
        #print("request_buy {}".format(buy))
        if (not self.bought):
            self.current = self.current - self.value
            self.holding = ask / self.value
            self.buy_price = ask
            if (self.verbose):
                print(f'{self.timestamp} Buy ({self.price}): ask: {ask}')
        self.bought = True
        
    def request_sell(self, sell):
        #print("best_sell {}".format(sell))
        if (not self.bought):
            return
        self.current = self.current + (sell / self.holding)
        self.holding = 0
        self.bought = False
        if (self.verbose):
            profit = (sell - self.buy_price)
            if (profit < 0):
                print(f'#### LOSSSS: {profit}')
            print(f'{self.timestamp} Sell ({self.price}) {profit} total: {self.current}')

    def on_state(self, timestamp, price):
        self.timestamp = pd.to_datetime(timestamp, unit='s')
        self.price = price
        #print(f'on_state: {self.price} {self.timestamp}')
        
    def report(self):
        print(self.initial_value)
        print(self.current)
        print(f'{((self.current*100)/self.initial_value) - 100}%')

class StockAgent():
    
    def __init__(self, model = [],
                request_sell = lambda price: price,
                request_buy = lambda price: price,
                on_state = lambda timestamp, price: timestamp
                ):
        self.model = model
        self.request_buy = request_buy
        self.request_sell = request_sell
        self.on_state = on_state
        self.best_sell = 0
        self.best_buy = 0
        self.timestamp = 0
        
    def on_x(self, x):
        y = self.model.predict(np.array([x]))
        self.on_predicted(y[0])
        
    def on_new_state(self, timestamp, price, bid, ask):
        self.best_ask = float(ask[-1][0])
        self.best_bid = float(bid[-1][0])
        self.timestamp = timestamp
        self.price = price
        self.on_state(timestamp, price)
        #print(self.best_buy)
        
    def on_predicted(self, y):
        #print(y)
        if (y > 0.5):
            self.buy()
        else:
            self.sell()
        
    def buy(self):
        #print(f'BUY: {self.price} best_ask : {self.best_ask} best_bid: {self.best_bid}')
        self.request_buy(self.price)
        #self.request_buy(self.best_ask)
        
    def sell(self):
        #print(f'SELL : {self.price} best_ask : {self.best_ask} best_bid: {self.best_bid}')
        self.request_sell(self.price)
        #self.request_sell(self.best_bid)
