from data_util import *
from agents.data_agent import *
from bitstamp import *
from entities.entities import *

import pandas as pd
import numpy as np
import datetime

class SourceDataGenerator():
    def __init__(self,
                 tec: TecAn,
                 base_dir = "data/"
                ):
        
        self.base_dir = base_dir
        self.tec: TecAn = tec

            
    def save(self, data_set, prefix = ""):
        trainX = data_set[0]
        trainY = data_set[1]
        final_path = self.base_dir
        train_path = "{}{}trainX.npy".format(final_path, prefix)
        np.save(train_path, trainX)
        np.save("{}{}trainY.npy".format(final_path, prefix), trainY)
        if (len(data_set) > 2):
            valX = data_set[2]
            valY = data_set[3]
            np.save("{}{}valX.npy".format(final_path, prefix), valX)
            np.save("{}{}valY.npy".format(final_path, prefix), valY)
        print("Saving {} with {}".format(train_path, trainX.shape))

    def get_y_data_old(self, ohlc, shift = -1):
        combined_data = ohlc.copy()
        #combined_data['return'] = np.log(combined_data / combined_data.shift(1))
        returns = (ohlc / ohlc.shift(shift))
        combined_data['return'] = returns
        combined_data['direction'] = np.where(combined_data['return'] < 1, 1, 0)
        #print(combined_data)
        #combined_data.dropna(inplace=True)
        #print(combined_data[20:40])
        #
        return combined_data['direction'].to_numpy()

    def split(self, x, y, split, shuffle=False):
        trainX, valX, trainY, valY = self.train_test_split(np.array(x), np.array(y), test_size=split, shuffle=shuffle)
        print("Completed: {} {} {} {}".format(trainX.shape, trainY.shape, valX.shape, valY.shape))
        return trainX, trainY, valX, valY


    def get_full_database(self, data_detail: DataDetail, raw_dir):

        full_data = self.base_dir + raw_dir + "/"
        data_gen = DataGenerator(random = False, base_dir = full_data)
        data_gen.rewind()
        data_count = (data_gen.steps - 100)
        #data_count = 200000

        final_x = []

        closed_prices = []

        on_new_data = lambda x: final_x.append(x)
        on_closed_price = lambda price: closed_prices.append(price)

        agent = DataAgent(
            tec = TecAn(windows = data_detail.windows),
            minutes = data_detail.minutes,
            on_new_data = on_new_data,
            on_closed_price = on_closed_price
        )

        print("Processing {}".format(raw_dir))

        for i in tqdm(range(data_count)):
            agent.on_new_raw_data(data_gen.next())


        closes = pd.DataFrame(closed_prices, columns = ['Close'])

        final_y = get_y_data(closes, -1)

        #print(agent.ohlc)

        return final_x, final_y, closed_prices


    def load_dataset(self, dir):
        self.load_datasets([dir])

    def load_datasets(self, dirs, resample):
        print(dirs)
        sets = []  
        for raw_dir in dirs:

            x, y, closed_prices = self.get_full_database(resample = resample,
                                                 raw_dir = raw_dir)

            final_data = self.split(x, y, 0.1, shuffle=False)

            self.save(final_data, raw_dir)
            sets.append((x, y))
        return sets

    def conc_sets(self, sets):
        trainX = sets[0][0]
        trainY = sets[0][1]
        for i in range(1,  len(sets)):
            data_set = sets[i]
            trainX = np.append(data_set[0], trainX, axis = 0)
            trainY = np.append(data_set[1], trainY, axis = 0)

        trainX, trainY, valX, valY = split(trainX, trainY, 0.1, shuffle=True)

        return trainX, trainY, valX, valY


    def load_simple_datasets(self, dirs, resample):
        print(f"Simple: {dirs}")
        sets = []  
        for raw_dir in dirs:

            x, y, closed_prices = self.get_full_database(resample = resample,
                                                 raw_dir = raw_dir)

            self.save((np.array(x), closed_prices), f"simple_{raw_dir}")
            sets.append((x, closed_prices))
        return sets

    def conc_simple_sets(self, sets):
        trainX = sets[0][0]
        trainY = sets[0][1]
        for i in range(1,  len(sets)):
            data_set = sets[i]
            trainX = np.append(data_set[0], trainX, axis = 0)
            trainY = np.append(data_set[1], trainY, axis = 0)

        return trainX, trainY
    
    
    def parse(self, parsed):
        for i in parsed: #observe all columns
            timestamp = datetime.datetime.fromtimestamp(int(i['timestamp']))
            #print(timestamp,i['high'],i['low'],i['open'],i['close'],i['volume'])

        # fill in a DF with the extracted data
        df = pd.DataFrame(parsed)
        return df

    def process_online_data(self, result, data_detail: DataDetail, currency, verbose):
        init_time_stamp = result[0]['timestamp']
        end_time_stamp = result[-1]['timestamp']
        init = datetime.datetime.fromtimestamp(int(init_time_stamp))
        end = datetime.datetime.fromtimestamp(int(end_time_stamp))

        if (verbose):
            print(f"Downloaded from {init}({init_time_stamp}) to {end}({end_time_stamp}) {result[-1]['open']}")

        OPEN = 'open'
        TIME = 'timestamp'
        VOLUME = 'volume'

        final_x = []
        prices = []
        timestamps = []

        on_new_data = lambda x: final_x.append(x)
        on_closed_price = lambda price: price

        agent = DataAgent(
            tec = TecAn(
                windows=data_detail.windows
            ),
            minutes = data_detail.minutes,
            on_new_data = on_new_data,
            on_closed_price = on_closed_price
        )

        if (verbose):
            print("Processing {} of {}".format(len(result), currency))

        result_list = result

        if (verbose):
            result_list = tqdm(result)

        for data in result_list:
            price = float(data[OPEN])
            volume = float(data[VOLUME])
            timestamp = int(data[TIME])
            prices.append(price)
            timestamps.append(timestamp)
            agent.on_new_price(timestamp, price, volume)

        #closes = pd.DataFrame(closed_prices, columns = ['Close'])

        #print(agent.ohlc)

        return np.array(final_x), np.array(prices), np.array(timestamps)

    def get_full_database_online(self, currency, data_detail: DataDetail, start, limit, verbose, end=-1):

        result = load_bitstamp_ohlc(currency, 
                                    start=start,
                                    end=end,
                                    step=data_detail.get_seconds(),
                                    limit=limit,
                                    verbose = verbose)

        return self.process_online_data(result = result, data_detail = data_detail, currency = currency, verbose = verbose)

    def get_full_database_online_period(self, currency, data_detail: DataDetail, start, end, verbose):
    
        result = load_bitstamp_ohlc_by_period(currency, 
                                    start=start,
                                    end=end,
                                    step=data_detail.get_seconds(),
                                    verbose = verbose
                                    )

        return self.process_online_data(result = result, data_detail = data_detail, currency = currency, verbose = verbose)

