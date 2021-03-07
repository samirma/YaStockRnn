import pandas as pd
from ta.trend import *
from ta.momentum import *
from ta.volume import *
from ta.volatility import *
from ta import add_all_ta_features, add_trend_ta, add_volume_ta, add_volatility_ta, add_momentum_ta, add_others_ta
import matplotlib.pyplot as plt


def ta_list(win):
    tas = []
    tas.append(lambda close, volume, current_close, current_volume: (wma_indicator(close)/current_close))
    tas.append(lambda close, volume, current_close, current_volume: ulcer_index(close)/current_close)
    tas.append(lambda close, volume, current_close, current_volume: bollinger_lband_indicator(close)/current_close)
    tas.append(lambda close, volume, current_close, current_volume: bollinger_hband_indicator(close)/current_close)
    tas.append(lambda close, volume, current_close, current_volume: bollinger_pband(close)/current_close)
    tas.append(lambda close, volume, current_close, current_volume: bollinger_wband(close)/current_close)
    tas.append(lambda close, volume, current_close, current_volume: bollinger_lband(close)/current_close)
    tas.append(lambda close, volume, current_close, current_volume: (bollinger_mavg(close)/current_close))
    tas.append(lambda close, volume, current_close, current_volume: (macd_diff(close)/current_close))
    tas.append(lambda close, volume, current_close, current_volume: (macd_signal(close)/current_close))
    tas.append(lambda close, volume, current_close, current_volume: (ema_indicator(close)/current_close))
    tas.append(lambda close, volume, current_close, current_volume: (stc(close, window_slow=win)))
    
    tas.append(lambda close, volume, current_close, current_volume: (kst_sig(close)))
    tas.append(lambda close, volume, current_close, current_volume: (aroon_down(close, window=win)))
    tas.append(lambda close, volume, current_close, current_volume: (aroon_up(close, window=win)))
    tas.append(lambda close, volume, current_close, current_volume: (volume_price_trend(close, volume)))
    tas.append(lambda close, volume, current_close, current_volume: (force_index(close, volume, window=win)))
    tas.append(lambda close, volume, current_close, current_volume: (negative_volume_index(close, volume)))
    return tas

class TecAn:
    
    def __init__(self, windows = 100, windows_limit = 200):
        self.windows = windows
        self.tas = ta_list(self.windows)
        self.data = []
        self.windows_limit = windows_limit

    #Process the raw state
    def add_ta(self, list, price, amount):
        self.data.append([price, amount])
        if (len(self.data) > self.windows_limit):
            self.data.pop(0)
        df = pd.DataFrame(self.data, columns = ['Close', 'Volume'])
        close = df['Close']
        volume = df['Volume']
        import math
        for ta in self.tas:
            value = ta(close, volume, price, amount).iloc[-1]
            #value = log(value)
            list.append(value)
        return list
        
