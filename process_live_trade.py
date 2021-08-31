from agents.data_agent import *
from agents.stock_agent import *
from joblib import *
from agents.tec_an import *
from bitstamp import *
from model_winner_select import *

import argparse

class RawStateDownloader(LiveBitstamp):
    
    def __init__(self, agent : DataAgent, 
                        stock : ModelAgent, 
                        back : BackTest, 
                        on_raw_data = lambda raw: print(raw), 
                        verbose = False):
        self.trade = {}
        self.verbose = verbose
        self.on_new_data_count = 0
        self.on_raw_data = on_raw_data
        self.agent = agent
        self.stock = stock
        self.back = back
        self.last_log = ""
                
    def process(self, raw):
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
        

def start_process_path(result_path, currency, simulate_on_price, hot_load):

    result = load(result_path) 

    start_process_by_result(result, currency, simulate_on_price, hot_load)

def start_process_index(results_path, index, currency, simulate_on_price, hot_load):
    
    result = load(results_path)

    print(f"Total: {len(result)}")

    start_process_by_result(result[index], currency, simulate_on_price, hot_load)

def init_raw_process(
    currency, 
    minutes, 
    agent :DataAgent, 
    model_agent :ModelAgent
    ):
    timestamp = int(datetime.timestamp((datetime.now())))

    step = minutes*60
    page = load_bitstamp_ohlc(
        currency_pair=currency,
        start=agent.last_timestamp,
        end=timestamp,
        step=step,
        limit=100,
        verbose=False
        )
    
    for data in page:
        open_price = float(data["open"])
        open_amount = float(data["volume"])
        open_timestamp = int(data["timestamp"])

        if (open_timestamp < agent.last_timestamp):
            continue

        recovered_date = pd.to_datetime(open_timestamp, unit='s')
        reference_date = pd.to_datetime(timestamp, unit='s')

        #print(f"Pre {agent.last_index}")
        agent.last_index = agent.process_data_input(
            price = open_price, 
            amount = open_amount, 
            timestamp = open_timestamp
            )
        #print(f"Pos {agent.last_index} -> {agent.last_timestamp}")
        #print(f"Init reference_date: {reference_date} recovered_date: {recovered_date} ")
    #print(f"Last index: {agent.last_index}")

def start_process_by_result(result: ModelDetail, currency, simulate_on_price, hot_load):
    model = result.model
    window = result.data_detail.windows
    minutes = result.data_detail.minutes
    step = result.data_detail.steps_ahead
    print(f"Minutes={minutes} Window={window} Step={step}")
    print(f"Simulate on price {simulate_on_price}")
    print(f"{model}")

    agent, back, stock = get_agent(minutes = minutes,
                                    win = window,
                                    step = step,
                                    currency = currency,
                                    hot_load = hot_load,
                                    model = model,
                                    timestamp = int(datetime.timestamp((datetime.now()))),
                                    simulate_on_price = simulate_on_price,
                                    verbose = True)

    back.verbose = True
    stock.verbose = True

    on_raw_data = lambda raw: agent.on_new_raw_data(raw)


    live = RawStateDownloader(
                        agent = agent,
                        stock = stock,
                        back = back,
                        on_raw_data = on_raw_data, 
                        verbose = False
                        )

    bt = Bitstamp(live, currency = currency)

    init_raw_process(currency, minutes, agent, stock)

    while (True):
        bt.connect()
        print("Reconnectiong")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--m', dest="results_path", action="store")
    parser.add_argument('--i', dest="index", action="store")

    parser.add_argument('--r', dest="result_path", action="store")
    parser.add_argument('--c', dest="currency", action="store", default="btcusd")

    parser.add_argument('--p', dest="simulate_on_price", default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument('--hot', dest="hot_load", default=True, action=argparse.BooleanOptionalAction)

    add_arguments_winner(parser)

    args = parser.parse_args()

    print(f"simulate_on_price --p: {args.simulate_on_price}")
    print(f"currency --c: {args.currency}")
    print(f"hot_load --hot: {args.hot_load}")
    print(f"use_trained_profit --use_trained_profit: {args.use_trained_profit}")

    if (args.result_paths_list != None and len(args.result_paths_list) > 0):
        print(f"minutes: {args.minutes_list}")
        print(f"Evaluate on currency_list: {args.currency_list}")
        print(f"Process on currency: {args.currency}")
        print(f"result_paths_list: {args.result_paths_list}")

        timestamp = int(datetime.timestamp((datetime.now())))
        winner = get_best_model(
            minutes_list=args.minutes_list,
            result_paths=args.result_paths_list,
            currency_list=args.currency_list,
            timestamp = timestamp,
            winner_path = None,
            use_trained_profit = args.use_trained_profit
        )

        start_process_by_result(
            result = winner, 
            currency = args.currency, 
            simulate_on_price = args.simulate_on_price, 
            hot_load = args.hot_load)
    elif (args.results_path != None and args.index != None):
        print(f"result_path --r: {args.result_path}")
        print(f"index --i: {args.index}")
        start_process_index(results_path = args.results_path, index = args.index, currency = args.currency, simulate_on_price = args.simulate_on_price)
    else:
        start_process_path(args.result_path, currency = args.currency, simulate_on_price = args.simulate_on_price)