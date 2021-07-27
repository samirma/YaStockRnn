import pandas as pd
from tec_an import TecAn
from data_agent import *
from data_util import *
from sklearn_model_hyper import *
import pandas as pd
from stock_agent import *
from backtest import *
from joblib import *
from tec_an import *
from bitstamp import *
from model import *
from datetime import datetime
import argparse

def load_online(minutes, window, val_end, currency = "btcusd"):
    tec = TecAn(windows = window, windows_limit = 100)
    source_data_generator = SourceDataGenerator(tec = tec)


    online = OnLineDataProvider(
                 source_data_generator = source_data_generator,
                 minutes = minutes,
                 train_keys = [],
                 train_limit = 40,
                 val_limit = 1000,
                 val_keys = [currency],
                 val_start = val_end - (60000 * 60),
                 val_end = val_end,
                 train_start_list = []
    )

    start = val_end - (60 * 400 * minutes)
    end = val_end - (60 * minutes)

    online.load_val_cache(
                    val_keys = [currency],                  
                    start = start,
                    end = end)
    return online


def get_agent(minutes, win, step, model, hot_load = True, currency = "btcusd"):
    
    back = BackTest(value = 100,
                        verbose = True,
                        pending_sell_steps = step, 
                        sell_on_profit = True)

    request_sell = lambda price: back.request_sell(price)
    request_buy = lambda price: back.request_buy(price)
    on_state = lambda timestamp, price: back.on_state(timestamp, price)

    stock = StockAgent(
        model = model,
        request_sell = request_sell,
        request_buy = request_buy,
        on_state = on_state
    )


    on_new_data = lambda x: print(x)
    on_new_data = lambda x: stock.on_x(x)

    on_state = lambda timestamp, price, buy, sell: print("{} {} {} {}".format(timestamp, price, buy, sell))
    on_state = lambda timestamp, price, buy, sell: stock.on_new_state(timestamp, price, buy, sell)

    agent = DataAgent(
        taProc = TacProcess(), 
        tec = TecAn(windows = win, windows_limit = 100),
        resample = f'{minutes}Min',
        on_state = on_state,
        on_new_data = on_new_data
    )
    
    if (hot_load):
        timestamp = int(datetime.timestamp((datetime.now())))
        online = load_online(minutes = minutes, 
                            currency= currency,
                             window = win, 
                            val_end = timestamp)
        valX, valY = online.load_val_data(currency)
        for yy in valY:
            agent.taProc.add_tacs_realtime([], yy, 0.0, agent.tec)
        eval_back, metrics = eval_step(model, currency, step, online)
        print("###### Past report ######")
        eval_back.report()
        print("###### - ######")

    return agent, back, stock


raw_data_live = []


class RawStateDownloader(LiveBitstamp):
    
    def __init__(self, agent : DataAgent, stock : StockAgent, back : BackTest, on_raw_data = lambda raw: print(raw), verbose = False):
        self.trade = {}
        self.verbose = verbose
        self.on_new_data_count = 0
        self.on_raw_data = on_raw_data
        self.agent = agent
        self.stock = stock
        self.back = back
        self.last_log = ""
                
    def process(self, raw):
        #raw_data_live.append(raw)
        self.on_raw_data(raw)
        #timestamp = raw[TIMESTAMP_KEY]
        #time = pd.to_datetime(timestamp, unit='s')
        if (self.verbose):
            log = f"{self.stock.get_last_action()} | {self.back.current}"
            if (log == self.last_log):
                return
            self.last_log = log
            print(f'{log}')
            #print(f'{datetime.now()}: {self.agent.on_new_data_count} {self.stock.get_last_action()} | {self.back.get_profit()}', end='\r')
        

def start_process_path(result_path, currency, simulate_on_price):

    result = load(result_path) 

    start_process_by_result(result, currency, simulate_on_price)

def start_process_index(results_path, index, currency, simulate_on_price):
    
    result = load(results_path)

    print(f"Total: {len(result)}")

    start_process_by_result(result[index], currency, simulate_on_price)

def start_process_by_result(result, currency, simulate_on_price):
    model = result['model']
    window = result['window']
    minutes = result['minutes']
    step = result['step']
    profit = result['profit']
    print(f"Minutes={minutes} Window={window} Step={step} | {profit}")
    print(f"Simulate on price {simulate_on_price}")
    print(f"{model}")

    agent, back, stock = get_agent(minutes = minutes,
                                    win = window,
                                    step = step,
                                    currency = currency,
                                    hot_load = True,
                                    model = model)

    stock.simulate_on_price = simulate_on_price

    on_raw_data = lambda raw: agent.on_new_raw_data(raw)


    live = RawStateDownloader(
                        agent = agent,
                        stock = stock,
                        back = back,
                        on_raw_data = on_raw_data, 
                        verbose = True
                        )

    bt = Bitstamp(live, currency = currency)

    bt.connect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--m', dest="results_path", action="store")
    parser.add_argument('--i', dest="index", action="store")

    parser.add_argument('--r', dest="result_path", action="store")
    parser.add_argument('--c', dest="currency", action="store", default="btcusd")

    parser.add_argument('--p', dest="simulate_on_price", default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if (args.results_path != None and args.index != None):
        start_process_index(results_path = args.results_path, index = args.index, currency = args.currency, simulate_on_price = args.simulate_on_price)
    else:
        start_process_path(args.result_path, currency = args.currency, simulate_on_price = args.simulate_on_price)