import pandas as pd
from ta.trend import *
from ta.momentum import *
from ta.volume import *
from ta.volatility import *
from ta import add_all_ta_features, add_trend_ta, add_volume_ta, add_volatility_ta, add_momentum_ta, add_others_ta
import matplotlib.pyplot as plt
import multiprocessing
import threading
import numpy as np


def ta_list(win, fillna=True):
    tas = []   
    #Trend
    tas.append(lambda close, volume, current_close, current_volume: (macd_diff(close, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (rsi(close, fillna=fillna)/100))
    tas.append(lambda close, volume, current_close, current_volume: (kst_sig(close, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (dpo(close, window=20, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (dpo(close, window=40, fillna=fillna)))

    tas.append(lambda close, volume, current_close, current_volume: (stochrsi(close, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (sma_indicator(close, fillna=fillna)/current_close))
    
    tas.append(lambda close, volume, current_close, current_volume: (np.log(aroon_down(close, fillna=fillna)/current_close)))
    tas.append(lambda close, volume, current_close, current_volume: (np.log(aroon_up(close, fillna=fillna)/current_close)))
    
    #Vol
    tas.append(lambda close, volume, current_close, current_volume: (ulcer_index(close, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (ulcer_index(close, window=30, fillna=fillna)))

    
    #Momentuom
    tas.append(lambda close, volume, current_close, current_volume: (tsi(close, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (stc(close, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (ppo(close, fillna=fillna)))
    #tas.append(lambda close, volume, current_close, current_volume: (kama(close, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (ppo_signal(close, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (pvo(close, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (roc(close, window=12, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (roc(close, window=30, fillna=fillna)))
    
    #Volume
    tas.append(lambda close, volume, current_close, current_volume: (on_balance_volume(close, volume, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (volume_price_trend(close, volume, fillna=fillna)))
    
    return tas



class TecAn:
    
    def __init__(self, windows = 30, windows_limit = 100):
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
    
    def generate_custom_ta(self, list, close):
        combined_data = close
        return_price = np.log(combined_data / combined_data.shift(1))
        
        self.add_custom_add(list, return_price.rolling(5).mean().shift(1))
        self.add_custom_add(list, return_price.rolling(20).std().shift(1))
        self.add_custom_add(list, (return_price - return_price.rolling(30).mean()).shift(1))
        
        self.add_custom_add(list, return_price.rolling(10).mean().shift(1))
        self.add_custom_add(list, return_price.rolling(10).std().shift(1))
        self.add_custom_add(list, (return_price - return_price.rolling(10).mean()).shift(1))
        
        

    #Process the raw state
    def add_ta(self, price, amount):
               
        list = []
        self.data.append([price, amount])
        if (len(self.data) > self.windows_limit):
            self.data.pop(0)
        
        df = pd.DataFrame(self.data, columns = ['Close', 'Volume'])
        close = df['Close']
        volume = df['Volume']
        #print("Received: {} {}".format(price, amount))
        for ta in self.tas:
            value = ta(close, volume, price, amount).iloc[-1]
            list.append(value)
        #print(list)
        
        self.generate_custom_ta(list, close)
        
        self.indicators = list
        self.price  = price
        self.amount = amount
        #print("new indices genereted")
        return list
        
