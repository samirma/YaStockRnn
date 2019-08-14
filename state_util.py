last_price = 0
last_time = 0
should_buy = 0
should_sell = 0

def get_date(state):
    timestamp = int(state["timestamp"])
    return datetime.fromtimestamp(timestamp)


def get_parse_state(raw_state, last_price, last_time):
    list = []
    price = raw_state["price"]
        
    list.append(raw_state["amount"])

    def prepare_orders(orders, price, multi):
        amount = float(orders[0][1])
        for order in orders:
            list.append((float(order[0])/price) * multi)
            list.append(float(order[1])/amount)

    history_step = 5
    bids = raw_state["bids"][:history_step]
    asks = raw_state["asks"][:history_step]
    prepare_orders(bids, price, 1)
    prepare_orders(asks, price, -1)

    if last_price != 0:
        list.extend([price/last_price])
    else:
        list.extend([0])
        
    current_timestamp = int(raw_state['timestamp'])
    if last_time != 0:
        list.extend([current_timestamp / last_time])
    else:
        list.extend([0])
    #list = []
    #list.extend([raw_state['timestamp']])
    #print(current_timestamp)
    return list

def get_future_state(df, sec=120): # 2 min in future
    state_timestamp = int(dt.get_from_index(dt.index-1)['timestamp'])
    timestamp_limit = state_timestamp + sec
    #print("Current timestamp", state_timestamp, " ==== ", timestamp_limit)
    index = 0
    timestamp = 0
    while timestamp < timestamp_limit:
        index += 3
        state = dt.get_from_index(dt.index + index)
        timestamp = int(state['timestamp'])
    return state



def get_state(raw_state):
    global last_price
    global last_time
    global should_buy
    global should_sell
    
    list = get_parse_state(raw_state, last_price, last_time)

    furure_state = get_future_state(dt)
    future_price = furure_state["price"]
    
    current_price = raw_state["price"]
    current_timestamp = int(raw_state['timestamp'])
    
    best_bid = float(furure_state["bids"][0][0]) 
    is_value_incresed = best_bid >= (current_price + 0.2)

    if is_value_incresed:
        should_buy += 1
        #print (current_price, " ==== ", (current_price + 0.2), " ===== ", furure_state)
        y = onehot_encoded(0)
    else:
        #print (current_price, " ==== ", (current_price + 0.2), " ===== ", furure_state)
        should_sell += 1
        y = onehot_encoded(1)
           
    #print (y)
    #print (get_date(raw_state), " ==== ", get_date(furure_state))
    
    last_price = current_price
    last_time = current_timestamp
    return [list, y]
