import pandas as pd
from ta.trend import *
from ta.momentum import *
from ta.volume import *
from ta.volatility import *
from ta import add_all_ta_features, add_trend_ta, add_volume_ta, add_volatility_ta, add_momentum_ta, add_others_ta
import matplotlib.pyplot as plt


def ta_list(win, fillna=True):
    tas = []   
    #tas.append(lambda close, volume, current_close, current_volume: (kst_sig(close, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (aroon_down(close, window=win, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (aroon_up(close, window=win, fillna=fillna)))
    #tas.append(lambda close, volume, current_close, current_volume: (volume_price_trend(close, volume, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (force_index(close, volume, window=win, fillna=fillna)))
    #tas.append(lambda close, volume, current_close, current_volume: (negative_volume_index(close, volume, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: ulcer_index(close, fillna=fillna))
    tas.append(lambda close, volume, current_close, current_volume: bollinger_lband_indicator(close, fillna=fillna))
    tas.append(lambda close, volume, current_close, current_volume: bollinger_hband_indicator(close, fillna=fillna))
    tas.append(lambda close, volume, current_close, current_volume: bollinger_pband(close, fillna=fillna))
    tas.append(lambda close, volume, current_close, current_volume: bollinger_wband(close, fillna=fillna))
    #tas.append(lambda close, volume, current_close, current_volume: bollinger_lband(close, fillna=fillna))
    #tas.append(lambda close, volume, current_close, current_volume: (bollinger_mavg(close, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (macd_diff(close, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (macd_signal(close, fillna=fillna)))
    tas.append(lambda close, volume, current_close, current_volume: (stc(close, window_slow=win, fillna=fillna)))

    return tas

class TecAn:
    
    def __init__(self, windows = 50, windows_limit = 50):
        self.windows = windows
        self.tas = ta_list(self.windows)
        self.data = []
        self.windows_limit = windows_limit
    
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
            #value = log(value)
            list.append(value)
        #print(list)
        return list
        
