from source_data_generator import *


class LocalDataProvider():
    
    def __init__(self, 
                train_keys,
                val_keys,
                path
                ):
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_data = []
        self.vals = {}
        self.path = path
    
    def load_train_data(self):
        return self.load_data("simple_full_", "train", self.path)
    
    def load_val_data(self, val):
        return self.load_data(f"simple_{val}", "train", self.path)


class OnLineDataProvider():
    #https://www.unixtimestamp.com/
    def __init__(self,
                 source_data_generator :SourceDataGenerator,
                 minutes,
                 val_start,
                 val_end,
                 train_keys,
                 val_keys,
                 verbose,
                 train_start_list,
                 train_limit = 100,
                 val_limit = 1000
                ):
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_data = []
        self.vals = {}
        self.minutes = minutes
        self.train_limit = train_limit
        self.val_limit = val_limit
        self.val_start = val_start
        self.val_end = val_end
        self.verbose = verbose
        self.train_start_list = train_start_list
        self.source_data_generator :SourceDataGenerator = source_data_generator
        self.data_detail: DataDetail = DataDetail(windows=source_data_generator.tec.windows, minutes=minutes, steps_ahead=1)
        self.resample = f'{minutes}Min'
    
    def windows(self):
        return self.data_detail.windows

    def load_cache(self):
        self.load_train_cache()
        self.load_val_cache(self.val_keys, self.val_start, self.val_end)

    def load_train_cache(self):
        sets = []

        def load_from_time(time): 
            for curr in self.train_keys:
                x, prices, times = self.source_data_generator.get_full_database_online(
                                                                                currency = curr, 
                                                                                data_detail = self.data_detail, 
                                                                                limit = self.train_limit,
                                                                                start = time,
                                                                                verbose = self.verbose
                                                                                )
                to_be_removed = 100
                sets.append((x[to_be_removed:], prices[to_be_removed:], times[to_be_removed:]))

        for start_time in self.train_start_list:
            load_from_time(start_time)

        if (len(sets)>0):
            self.train_data = self.source_data_generator.conc_simple_sets(sets)
            
    def load_val_cache(self, val_keys, start, end):

        for key in val_keys:
            x, prices, times = self.source_data_generator.get_full_database_online_period(
                                                                            currency = key,
                                                                            data_detail = self.data_detail,
                                                                            start=start,
                                                                            end=end,
                                                                            verbose = self.verbose
                                                                           )
            #to_be_removed = 100
            #self.vals[key] = (x[to_be_removed:], prices[to_be_removed:], times[to_be_removed:])
            self.vals[key] = x, prices, times
            
    def load_train_data(self):
        return self.train_data
    
    def load_val_data(self, val):
        return self.vals[val]
    
    def report(self):
        print(f"Total train set {len(self.train_data[0])}")
        for key in self.val_keys:
            print(f"Total val {key} set {len(self.vals[key][0])}")

    def __str__(self):
        resume = {}
        for val in self.val_keys:
            resume[val] = len(self.vals[val])
        return f"OnLineDataProvider (val_keys = {resume}, train_data = {len(self.train_data)} , data_detail = {self.data_detail})"
