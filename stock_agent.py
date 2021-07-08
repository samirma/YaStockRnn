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
                 verbose = False,
                 pending_sell_steps = 1,
                 sell_on_profit = True
                ):
        self.initial_value = initial_value
        self.value = value
        self.verbose = verbose
        self.pending_sell_steps = pending_sell_steps
        self.pending = -1
        self.sell_on_profit = sell_on_profit
        self.reset()
        
        
    def reset(self):
        self.current = self.initial_value
        self.holding = 0
        self.buy_price = 0
        self.timestamp = 0
        
        
    def on_state(self, timestamp, price):
        self.timestamp = pd.to_datetime(timestamp, unit='s')
        self.price = price
        self.pending -= 1
        #print(f'on_state: {self.price} {self.timestamp}')
       
    def is_sell_pending(self):
        return (self.pending >= 1)

    def is_bought(self):
        return (self.holding > 0)
    
    def is_valid_sell(self, bid):
        is_pending = self.is_sell_pending()
        
        has_loss = (bid < self.buy_price)
        
        is_not_pending = (not is_pending)
        
        profit_before_pending = ((is_pending and (not has_loss)) and self.sell_on_profit)

        pending_finished_should_sell = (is_not_pending and self.is_bought())
        
        is_valid = (self.is_bought() and (is_not_pending or profit_before_pending))
        
        #print(f"{is_valid} pending ({self.pending}): {is_pending} - {is_not_pending} - {profit_before_pending}")
        
        return is_valid
     
    def request_buy(self, ask):
        #print("request_buy {}".format(buy))
        if (not self.is_bought()):
            self.buy(ask)
        self.pending = self.pending_sell_steps
        
    def request_sell(self, bid):
        if (self.is_valid_sell(bid)):
            self.sell(bid)
            self.pending = -1
        
        
    def report(self):
        print(self.initial_value)
        print(self.current)
        percentage = self.get_profit()
        print(f'{percentage}%')
        
    def get_profit(self):
        percentage = ((self.current*100)/self.initial_value) - 100
        return round(percentage, 4)
        
    def buy(self, ask):
        self.current = self.current - self.value
        self.holding = self.value / ask
        self.buy_price = ask
        if (self.verbose):
            print(f'{self.timestamp} Buy ({self.price}): ask: {ask}')
 
    def sell(self, sell):
        self.current = self.current + (sell * self.holding)
        if (self.verbose):
            profit = (sell - self.buy_price) * self.holding
            profit = round(profit, 4)
            if (profit < 0):
                print(f'#### LOSSSS: {profit}')
            print(f'{self.timestamp} Sell ({self.price}) profit: {profit} total: {self.current}')
        self.holding = 0
        self.buy_price = 0
            

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
