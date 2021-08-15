import pandas as pd
import datetime as dt
import numpy as np
from sklearn_model_hyper import *
import pandas as pd
from pandas import concat

class BackTest():
    
    def __init__(self, 
                 pending_sell_steps,
                 initial_value = 100, 
                 value = 100,
                 verbose = False,
                 sell_on_profit = True
                ):
        self.initial_value = initial_value
        self.value = value
        self.verbose = verbose
        self.pending_sell_steps = pending_sell_steps
        self.pending = -1
        self.sell_on_profit = sell_on_profit
        self.reset()
        if (self.verbose):
            print(self)
        
    def reset(self):
        self.current = self.initial_value
        self.holding = 0
        self.buy_price = 0
        self.positive_trades = 0
        self.negative_trades = 0
        
        
    def on_state(self):
        self.pending -= 1
       
    def is_sell_pending(self):
        return (self.pending >= 1)

    def is_bought(self):
        return (self.holding > 0)
    
    def is_valid_sell(self, bid):
        is_pending = self.is_sell_pending()
        
        has_profit = (bid > self.buy_price)
        
        is_not_pending = (not is_pending)
        
        profit_before_pending = ((is_pending and (has_profit)) and self.sell_on_profit)
        
        is_valid = (self.is_bought() and (is_not_pending or profit_before_pending))
        
        #self.log(f"{is_valid} Pending {self.pending} Bid ({bid}): is_bought: {self.is_bought()} - is_not_pending: {is_not_pending} - profit_before_pending: {profit_before_pending}")
        
        return is_valid
     
    def on_up(self, bid, ask):
        self.on_state()
        if (self.is_bought()):
            positive, profit = self.is_profit(bid)
            if (positive):
                self.log(f"Profit detected bid: {bid} ask: {ask}")
                self.sell(bid)
        else:
            self.buy(ask)
        
    def on_down(self, bid, ask):
        self.on_state()
        if (self.is_valid_sell(bid)):
            self.sell(bid)
            self.pending = -1
        
        
    def report(self):
        percentage = self.get_profit()
        print(f'{percentage}% -> {self.current}')
        print(f'Positive: {self.positive_trades} Negative: {self.negative_trades}')
        
    def get_profit(self):
        percentage = ((self.current*100)/self.initial_value) - 100
        #return self.positive_trades - self.negative_trades
        return round(percentage, 5)
        
    def buy(self, ask):
        self.current = self.current - self.value
        self.holding = self.value / ask
        self.buy_price = ask
        self.pending = self.pending_sell_steps
        self.log(f'Bought: {ask}')

    def is_profit(self, bid):
        profit = (bid - self.buy_price) * self.holding
        profit = round(profit, 4)
        positive = (profit > 0)
        return positive, profit
 
    def sell(self, bid):
        self.current = self.current + (bid * self.holding)
        positive, profit = self.is_profit(bid)

        if (positive):
            self.positive_trades += 1
            result = f"PROFIT {profit}"
        else:
            self.negative_trades += 1
            result = f"LOSS {profit}"
        
        self.log(f'SOLD >>>> Result: {result} total: {self.current}')
        self.holding = 0
        self.buy_price = 0

    def log(self, message):
        if (self.verbose):
            print(f'{dt.datetime.now()} BackTest: {message}')
    
    def __str__(self) -> str:
        return f"BackTest (pending_sell_steps={self.pending_sell_steps} sell_on_profit={self.sell_on_profit} value={self.value})"
            

class ModelAgent():
    
    def __init__(self, model = [],
                on_down = lambda bid, ask: bid,
                on_up = lambda bid, ask: ask,
                verbose = False,
                simulate_on_price = True,
                save_history = False
                ):
        self.model = model
        self.on_up = on_up
        self.on_down = on_down
        self.best_sell = 0
        self.best_buy = 0
        self.timestamp = 0
        self.price = 0
        self.verbose = verbose
        self.simulate_on_price = simulate_on_price
        self.last_action = {}
        self.last_action["time"] = ''
        self.last_action["action"] = ''
        self.history = []
        self.save_history = save_history
        
    def on_x(self, x):
        y = self.model.predict(np.array([x]))
        return self.on_predicted(y[0])
            
        
    def on_new_state(self, timestamp, price, bid, ask):
        self.best_ask = float(ask[-1][0])
        self.best_bid = float(bid[-1][0])
        self.timestamp = timestamp
        self.price = price
        #print(self.best_buy)
        
    def on_predicted(self, y):
        is_up = y > 0.5
        if (is_up):
            self.up()
        else:
            self.down()
        return is_up
        
    def up(self):
        if (self.simulate_on_price):
            self.log_action(f'UP')
            self.on_up(ask = self.price, bid = self.price)
        else:
            self.log_action(f'UP on ask: {self.best_ask}')
            self.on_up(ask = self.best_ask, bid = self.best_bid)
        
    def down(self):
        if (self.simulate_on_price):
            self.log_action(f'DOWN')
            self.on_down(ask = self.price, bid = self.price)
        else: 
            self.log_action(f'DOWN on bid: {self.best_bid}')
            self.on_down(ask = self.best_ask, bid = self.best_bid)

    def log_action(self, action):
        self.last_action["time"] = dt.datetime.now()
        self.last_action["action"] = action
        stock_date = pd.to_datetime(self.timestamp, unit='s')
        if (self.verbose):
            print(f"{dt.datetime.now()} ModelAgent({self.price}): {stock_date}({self.timestamp}) {self.get_last_action()}")

    def get_last_action(self):
        return f"{self.last_action['action']}"


def series_to_supervised(data, n_in=2, n_out=1):
    n_vars = 1
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    
    return agg.fillna(0).values