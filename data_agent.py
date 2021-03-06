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
from tec_an import TecAn


class TacProcess():
    def __init__(self):
        self.old_price = -1
        
    def add_tacs(self, list, index, result):
        list = []
        #print(len(result[-1]))
        #print(index)
        for tec in result:
            list.append(tec.iloc[index])
        return list

    def add_tacs_realtime(self, list, price, amount, tec):
        list = []
        list.extend(tec.add_ta(price, amount))
        self.old_price = price
        #print("{} {} {}".format(price, amount, list))
        return list
    

TIMESTAMP_KEY = "timestamp"
MICROTIMESTAMP_KEY = "microtimestamp"
ASKS_KEY = "asks"
BIDS_KEY = "bids"
PRICE_KEY = "price"
AMOUNT_KEY = "amount"
CLOSE = 'close'

class DataAgent():
    
    def __init__(self,
                 resample,
                 taProc = TacProcess(), 
                 tec = TecAn(windows = 20, windows_limit = 100),
                 on_new_data = lambda x: print("{}".format(x)),
                 on_state = lambda timestamp, price, bid, ask: price,
                 on_closed_price = lambda price: price
                 ):
        self.taProc = taProc
        self.tec = tec
        self.final_x = []
        self.list = []
        self.resample = resample
        self.raw_limit = 10000
        self.last_price = -1 
        self.last_amount = -1
        self.last_ohlc_count = 1
        self.on_new_data = on_new_data
        self.on_state = on_state
        self.on_closed_price = on_closed_price
        self.on_new_data_count = 0
        print("Resample {} - {}".format(self.resample, self.tec))
        
    def on_new_data(self, x):
        self.on_new_data(x)
        
        
    def on_new_raw_data(self, raw):
        price = raw[PRICE_KEY]
        amount = raw[AMOUNT_KEY]
        timestamp = raw[TIMESTAMP_KEY]
        
        self.on_state(timestamp, price, raw[BIDS_KEY], raw[ASKS_KEY])

        # Only consider when prices changes
        if (self.last_price == price and self.last_amount == amount):
            return
        
        self.last_price = price 
        self.last_amount = amount
        
        timestamp = pd.to_datetime(timestamp, unit='s')
        self.list.append([timestamp, price])
        
        if (len(self.list) > self.raw_limit):
            self.list.pop(0)
        
        DATE = 'Date'
        df = pd.DataFrame(self.list, columns = [DATE, CLOSE])
        df = df.set_index(pd.DatetimeIndex(df[DATE]))
                
        time = df[CLOSE].resample(self.resample)
        ohlc = time.ohlc()
        
        #print("{} {}".format(timestamp, len(ohlc)))
        
        ohlc_count = len(ohlc)
        if (ohlc_count < 2 or self.last_ohlc_count == ohlc_count):
            return
        
        self.last_ohlc_count = ohlc_count
        
        #print("{} - {}".format(self.last_ohlc_count, ohlc_count))
        
        #del ohlc['open']
        del ohlc['high']
        del ohlc['low']
        
        #price_closed = ohlc.iloc[-2][CLOSE]
        #print("######")
        #print(f'{price} open: {ohlc.iloc[-1]["open"]} price_closed: {price_closed}')
        #print(ohlc)
        
        self.on_new_data_count = self.on_new_data_count + 1
        
        self.on_new_price(price)
        
        self.ohlc = ohlc
        

    def on_new_price(self, price):
        self.on_closed_price(price)
        
        x = self.taProc.add_tacs_realtime([], price, 0.0, self.tec)
        self.on_new_data(x)
        
        
