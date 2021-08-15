import pandas as pd
from ta.trend import *
from ta.momentum import *
from ta.volume import *
from ta.volatility import *
from ta import add_all_ta_features, add_trend_ta, add_volume_ta, add_volatility_ta, add_momentum_ta, add_others_ta
import numpy as np


def ta_list(win, fillna=True):
    tas = []   
    #Trend
    tas.append(lambda close, volume, current_close, current_volume: (macd_diff(close, window_fast=win, window_slow=(win*2), fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (rsi(close, window=win, fillna=fillna)/100))
    tas.append(lambda close, volume, current_close, current_volume: (kst_sig(close, fillna=fillna)))
    #tas.append(lambda close, volume, current_close, current_volume: (dpo(close, window=win, fillna=fillna)))

    #tas.append(lambda close, volume, current_close, current_volume: (stochrsi(close, window=win, fillna=fillna)))
    #tas.append(lambda close, volume, current_close, current_volume: (sma_indicator(close, window=win, fillna=fillna)/current_close))
    
    #tas.append(lambda close, volume, current_close, current_volume: (aroon_down(close, window=win, fillna=fillna)))
    #tas.append(lambda close, volume, current_close, current_volume: (aroon_up(close, window=win, fillna=fillna)))
    
    #Vol
    #tas.append(lambda close, volume, current_close, current_volume: (ulcer_index(close, fillna=fillna)))
    #tas.append(lambda close, volume, current_close, current_volume: (ulcer_index(close, window=30, fillna=fillna)))

    
    #Momentuom
    tas.append(lambda close, volume, current_close, current_volume: (tsi(close, window_fast=win, window_slow=(win*2), fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (stc(close, window_fast=win, window_slow=(win*2), fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (ppo(close, fillna=fillna)))
    #tas.append(lambda close, volume, current_close, current_volume: (kama(close, fillna=fillna)))
    #tas.append(lambda close, volume, current_close, current_volume: (ppo_signal(close, window_fast=win, window_slow=(win*2), fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (pvo(close, window_fast=win, window_slow=(win*2), fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (roc(close, window=win, fillna=fillna)))
    
    #Volume
    #tas.append(lambda close, volume, current_close, current_volume: (on_balance_volume(close, volume, fillna=fillna)))
    #tas.append(lambda close, volume, current_close, current_volume: (volume_price_trend(close, volume, fillna=fillna)))
    
    return tas



class TecAn:
    
    def __init__(self, 
                 windows = 30, 
                 windows_limit = 100):
        self.windows = windows
        self.tas = ta_list(self.windows)
        self.data = []
        self.price = 0
        self.amount = 0
        self.indicators = []
        self.windows_limit = windows_limit
    
    def method(self, ta, close, volume, price, amount, results, index):
        #print("Starting {}".format(ta))
        value = ta(close, volume, price, amount).iloc[-1]
        #print("Stoping {}".format(ta))
        results[index] = value
    
    def add_custom_add(self, list, df):
        value = df.iloc[-1]
        if (np.isnan(value)):
            value = 0
        list.append(value)
    
    def generate_custom_ta(self, list, close, win):
        combined_data = close
        return_price = np.log(combined_data / combined_data.shift(1))
        
        self.add_custom_add(list, return_price.rolling(win).mean().shift(1))
        self.add_custom_add(list, return_price.rolling(win*2).std().shift(1))
        self.add_custom_add(list, (return_price - return_price.rolling(win).mean()).shift(1))
        
        self.add_custom_add(list, return_price.rolling(win).mean().shift(1))
        self.add_custom_add(list, return_price.rolling(win).std().shift(1))
        self.add_custom_add(list, (return_price - return_price.rolling(win).mean()).shift(1))
        
        

    #Process the raw state
    def add_ta(self, price, amount):
        list = []
        self.data.append([price, amount])
        if (len(self.data) > self.windows_limit):
            self.data.pop(0)
        
        df = pd.DataFrame(self.data, columns = ['Close', 'Volume'])
        close = df['Close']
        volume = df['Volume']
        #print("Received: {} {}".format(type(price), type(amount)))

        self.generate_custom_ta(list, close, self.windows)
        
        for ta in self.tas:
            value = ta(close, volume, price, amount).iloc[-1]
            if (np.isnan(value)):
                value = 0
            list.append(value)
        #print(list)
                
        self.indicators = list
        self.price  = price
        self.amount = amount
        #print("new indices genereted")
        return list
    
    def __str__(self):
        return "TecAn ( windows %s, windows_limit %s )" % (self.windows, self.windows_limit)