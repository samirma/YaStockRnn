from data_util import load_val_data, load_val_data_with_total
import datetime as dt

class CacheProvider():
    def __init__(self, currency_list, verbose):
        self.cache = {}
        self.agent_cache = {}
        self.currency_list = currency_list
        self.verbose = verbose

    def get_provider(self, minutes, windows, val_start, val_end):
            cache_key = f"{minutes}-{windows}"
            try:
                online = self.cache[cache_key]
                self.log(f"Loaded: {cache_key}")
            except:
                online = load_val_data(
                    minutes = minutes,
                    window = windows,
                    val_start = val_start,
                    val_end = val_end,
                    verbose = False,
                    currency_list = self.currency_list
                    )
                self.cache[cache_key] = online
                self.log(f"Added: {cache_key}")

            return online

    def get_provider_total(self, minutes, windows, total, val_end):
        cache_key = f"{minutes}-{windows}-{total}-{val_end}"
        try:
            online = self.cache[cache_key]
            self.log(f"Loaded: {cache_key}")
        except :
            online = load_val_data_with_total(minutes = minutes,
                                    window = windows,
                                    total = 200, 
                                    currency_list = self.currency_list,
                                    verbose = False,
                                    val_end = val_end)

            self.cache[cache_key] = online
            self.log(f"Added: {cache_key}")

        return online

    def log(self, message):
        if (self.verbose):
            print(f'{dt.datetime.now()} CacheProvider: {message}')