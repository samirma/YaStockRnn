from datetime import datetime

class StateUtil():
    
    def __init__(self, data_gen, future=5, on_state_parsed = lambda list, price, amount, index: list): # 1 min in future
        self.should_buy = 0
        self.should_sell = 0
        self.TIMESTAMP_KEY = "timestamp"
        self.MICROTIMESTAMP_KEY = "microtimestamp"
        self.ASKS_KEY = "asks"
        self.BIDS_KEY = "bids"
        self.PRICE_KEY = "price"
        self.AMOUNT_KEY = "amount"
        self.on_state_parsed = on_state_parsed
        self.future = future * 1000000
        self.data_gen = data_gen
        self.book_size = 10
        self.last_price = 0
    
    # integer encode input data
    def onehot_encoded (self, integer_encoded, char_to_int = 2):
        # one hot encode
        onehot_encoded = list()
        letter = [0 for _ in range(char_to_int)]
        letter[integer_encoded] = 1
        onehot_encoded.append(letter)

        return onehot_encoded[0]

    def get_date(self, state):
        timestamp = int(state[self.TIMESTAMP_KEY])
        return datetime.fromtimestamp(timestamp)


    def get_parse_state(self, raw_state):
        list = []
        price = raw_state[self.PRICE_KEY]
        amount = raw_state[self.AMOUNT_KEY]
        
        def prepare_orders(orders, price):
            for order in orders:
                list.append((float(order[0])/price) - 1)
                #list.append((float(order[1])/amount) - 1)

        history_step = self.book_size
        bids = raw_state[self.BIDS_KEY][:history_step]
        asks = raw_state[self.ASKS_KEY][:history_step]
        prepare_orders(bids, price)
        prepare_orders(asks, price)
        
        list.append(((self.last_price/price)-1))
        
        return self.on_state_parsed(list, price, amount, self.data_gen.index)

    def get_future_state(self, current_timestamp, current_index):
        sec=self.future
        timestamp_limit = current_timestamp + sec
        timestamp = 0
        #print("Current timestamp", state_timestamp, " ==== ", timestamp_limit)
        index = 1
        while True:
            state = self.data_gen.get_from_index(current_index + index)
            timestamp = int(state[self.MICROTIMESTAMP_KEY])
            #print("Current timestamp", timestamp, " ==== ", timestamp_limit)
            if (timestamp >= timestamp_limit):
                return state
            index += 100
            #print("Searching {}".format(timestamp_limit + index))
        return None

    
    def get_bid_goal(self, ask):
        return (ask * 1.0001)

    def get_state(self, raw_state, index):

        x = self.get_parse_state(raw_state)
        
        current_timestamp = int(raw_state[self.MICROTIMESTAMP_KEY])

        furure_state = self.get_future_state(current_timestamp, index)
        future_price = furure_state[self.PRICE_KEY]

        current_price = raw_state[self.PRICE_KEY]
        
        current_bid = float(raw_state[self.BIDS_KEY][0][0])
        future_bid = float(furure_state[self.BIDS_KEY][0][0])

        ask = float(raw_state[self.ASKS_KEY][0][0]) 
        predicted = self.get_bid_goal(ask)
        target_rate = ((future_bid/ask) - 1)
        is_value_incresed = (target_rate > 0)

        if is_value_incresed:
            self.should_buy += 1
            #print (self.get_date(raw_state), ": ", ask, " ==== ", predicted, " ===== ", self.get_date(furure_state), " ===== ", future_bid)
            #print(raw_state)
            #print(furure_state)
            #print("=====")
            y = self.onehot_encoded(1)
        else:
            #print (current_price, " ==== ", (current_price + 0.2), " ===== ", furure_state)
            self.should_sell += 1
            y = self.onehot_encoded(0)

        #y = target_rate
        #print (y)
        #print (get_date(raw_state), " ==== ", get_date(furure_state))

        self.last_price = current_price
        self.last_time = current_timestamp
        return [x, y, raw_state, furure_state]