from datetime import datetime

class StateUtil():
    
    def __init__(self):
        self.should_buy = 0
        self.should_sell = 0
        self.TIMESTAMP_KEY = "timestamp"
        self.ASKS_KEY = "asks"
        self.BIDS_KEY = "bids"
        self.PRICE_KEY = "price"
        self.AMOUNT_KEY = "amount"
    
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

        list.append(raw_state[self.AMOUNT_KEY])

        def prepare_orders(orders, price, multi):
            amount = float(orders[0][1])
            for order in orders:
                list.append((float(order[0])/price) * multi)
                list.append(float(order[1])/amount)

        history_step = 5
        bids = raw_state[self.BIDS_KEY][:history_step]
        asks = raw_state[self.ASKS_KEY][:history_step]
        prepare_orders(bids, price, 1)
        prepare_orders(asks, price, -1)

        #if last_price != 0:
        #    list.extend([price/last_price])
        #else:
        #    list.extend([0])

        #current_timestamp = int(raw_state[self.TIMESTAMP_KEY])
        #if last_time != 0:
        #    list.extend([current_timestamp / last_time])
        #else:
        #    list.extend([0])
        #list = []
        #list.extend([raw_state['timestamp']])
        #print(current_timestamp)
        return list

    def get_future_state(self, data_gen, sec=120): # 2 min in future
        state_timestamp = int(data_gen.get_from_index(data_gen.index-1)[self.TIMESTAMP_KEY])
        
        timestamp_limit = state_timestamp + sec
        #print("Current timestamp", state_timestamp, " ==== ", timestamp_limit)
        index = 0
        timestamp = 0
        while timestamp < timestamp_limit:
            state = data_gen.get_json_from_timestamp(timestamp_limit + index)
            if (state):
                timestamp_found = int(state[self.TIMESTAMP_KEY])
                #print("Current timestamp", state_timestamp, " ==== ", timestamp_found)
                return state
            index += 1
        return False

    
    def get_bid_goal(self, ask):
        return (ask * 1.0001)

    def get_state(self, raw_state, data_gen):

        list = self.get_parse_state(raw_state)

        furure_state = self.get_future_state(data_gen)
        future_price = furure_state[self.PRICE_KEY]

        current_price = raw_state[self.PRICE_KEY]
        current_timestamp = int(raw_state[self.TIMESTAMP_KEY])

        current_bid = float(raw_state[self.BIDS_KEY][0][0])
        future_bid = float(furure_state[self.BIDS_KEY][0][0])

        will_offer_less = (future_bid/current_bid) < 0.99998

        ask = float(raw_state[self.ASKS_KEY][0][0]) 
        predicted = self.get_bid_goal(ask)
        is_value_incresed = future_bid >= predicted

        if is_value_incresed:
            self.should_buy += 1
            print (self.get_date(raw_state), ": ", ask, " ==== ", predicted, " ===== ", self.get_date(furure_state), " ===== ", future_bid)
            #print(raw_state)
            #print(furure_state)
            #print("=====")
            y = self.onehot_encoded(0)
        else:
            #print (current_price, " ==== ", (current_price + 0.2), " ===== ", furure_state)
            self.should_sell += 1
            y = self.onehot_encoded(1)

        #print (y)
        #print (get_date(raw_state), " ==== ", get_date(furure_state))

        last_price = current_price
        last_time = current_timestamp
        return [list, y]
