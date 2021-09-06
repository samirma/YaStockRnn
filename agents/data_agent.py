import pandas as pd
from pandas._libs.tslibs import timestamps
from ta.trend import *
from ta.momentum import *
from ta.volume import *
from ta.volatility import *
from agents.tec_an import TecAn
import datetime as dt


TIMESTAMP_KEY = "timestamp"
MICROTIMESTAMP_KEY = "microtimestamp"
ASKS_KEY = "asks"
BIDS_KEY = "bids"
PRICE_KEY = "price"
AMOUNT_KEY = "amount"
CLOSE = 'close'
DATE = 'Date'

class DataAgent():
    
    def __init__(self,
                 minutes,
                 tec,
                 on_new_data = lambda x: print("{}".format(x)),
                 on_state = lambda timestamp, price, bid, ask: price,
                 on_closed_price = lambda price: price,
                 verbose = False,
                 save_history = False,
                 ):
        self.tec :TecAn = tec
        self.final_x = []
        self.list = []
        self.minutes = minutes
        self.raw_limit = 10000
        self.last_price = -1 
        self.last_amount = -1
        self.last_timestamp = -1
        self.last_processed_index = -1
        self.on_new_data = on_new_data
        self.on_state = on_state
        self.on_closed_price = on_closed_price
        self.verbose = verbose
        self.history = []
        self.save_history = save_history
        if (self.verbose):
            print("DataAgent (resample: {self.resample} tec: {self.tec})")
        
    def on_new_data(self, x):
        self.on_new_data(x)

    def on_new_raw_data(self, raw):
        price = raw[PRICE_KEY]
        amount = raw[AMOUNT_KEY]
        timestamp = int(raw[TIMESTAMP_KEY])
        bids = raw[BIDS_KEY]
        asks = raw[ASKS_KEY]
        self.process_data(
            price = price, 
            amount = amount, 
            timestamp = timestamp, 
            bids = bids, 
            asks = asks
            )
        
    def resample(self):
        return f'{self.minutes}Min'

    def process_data(self, price, amount, timestamp, bids, asks):
        
        self.on_state(timestamp, price, bids, asks)

        # Only consider when prices changes
        if (self.last_price == price and self.last_amount == amount and self.last_timestamp == timestamp):
            return

        current_index = self.process_data_input(price, amount, timestamp)

        index_log = f"last_index: {self.last_processed_index} current_index: {current_index}"

        if (self.last_processed_index == current_index):
            #self.log(f"Returning {index_log}")
            return self.last_processed_index

        #self.log(f"{index_log}")

        if (current_index == None):
            raise SystemExit(f"{price}, {amount}, {timestamp}")

        self.validate_data(current_index)

        self.list = self.list[-1:]

        self.update_index(current_index)

        self.on_new_price(
            timestamp=current_index.timestamp(),
            price=price, 
            amount=amount
            )


    def validate_data(self, current_index):
        if (self.last_processed_index != -1):
            timeframe = self.minutes * 60
            #print(current_index)
            #print(self.last_index)
            current_timestamp = int(current_index.timestamp())
            last_timestamp = int(self.last_processed_index.timestamp())
            if (current_timestamp < last_timestamp):
                raise SystemExit(f"last_index: {self.last_processed_index}({last_timestamp}) current_index: {current_index}({current_timestamp})")

            self.check_consistency( 
                                current_index = current_index, 
                                last_index = self.last_processed_index, 
                                timeframe = timeframe, 
                                tag = "AGENT"
                                )

            if (self.tec.last_index != self.last_processed_index):
                raise SystemExit(f"Invalid indexes Tec.last_index: {self.tec.last_index} last_index: {self.last_processed_index}")


    def update_index(self, current_index):
        self.last_processed_index = current_index

    def check_consistency(self, 
                        current_index,
                        last_index,
                        timeframe,
                        tag):
        current_timestamp = int(current_index.timestamp())
        last_timestamp = int(last_index.timestamp())
        diff = (current_timestamp - last_timestamp)
        if (diff != timeframe):
            error_msg = f"{tag} Diff {diff} Timeframe: {timeframe} last_index: {last_index}({last_timestamp}) current_index: {current_index}({current_timestamp})"
            raise SystemExit(error_msg)
        
    def process_data_input(self, price, amount, timestamp):
        #print(f"{self.last_timestamp} -> {self.last_price} {amount}")
        self.last_price = price 
        self.last_amount = amount
        self.last_timestamp = timestamp
        timestamp_pd = pd.to_datetime(timestamp, unit='s')
        self.list.append([timestamp_pd, price])

        df = pd.DataFrame(self.list, columns = [DATE, CLOSE])
        df = df.set_index(pd.DatetimeIndex(df[DATE]))
                
        time = df[CLOSE].resample(self.resample())
        ohlc = time.ohlc()
        
        self.ohlc = ohlc

        current_index = ohlc.index[-1]

        return current_index

    
    def on_action(self, action):
        if (self.save_history):
            self.history.append(action)

    def on_new_price(self, timestamp, price, amount):
        self.on_closed_price(price)
        x = self.tec.add_tacs_realtime(
            list = [],
            price = price, 
            amount = amount, 
            timestamp = timestamp
            )
        is_up = self.on_new_data(x)
        action = AgentHistory(
                timestamp = timestamp,
                price = price,
                x = x,
                is_up = is_up
        )
        self.on_action(action)
         
    def report(self):
        for data in self.history:
            print(f"{data.timestamp} - {data.price} - {data.is_up}")

    def log(self, message):
        if (self.verbose):
            print(f'{dt.datetime.now()} DataAgent: {message}')


class AgentHistory():
    
    def __init__(self,
                 timestamp, 
                 price, 
                 x, 
                 is_up
                 ):
        self.timestamp = timestamp
        self.price = price
        self.x = x
        self.is_up = is_up

    def __str__(self) -> str:
        return f"AgentHistory (timestamp={self.timestamp} price={self.price} is_up={self.is_up})"
           