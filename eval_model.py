from providers import OnLineDataProvider
from agents.data_agent import *
from agents.stock_agent import *
from data_util import *
from providers import *
from cache_providers import *
from sklearn.metrics import *
from collections import Counter

def add_hot_load(minutes, 
                win, 
                total, 
                currency, 
                timestamp_end, 
                verbose, 
                back: BackTest, 
                model_agent: ModelAgent, 
                agent: DataAgent,
                cache: CacheProvider
                ):
                
    model_agent.verbose = False
    back.verbose = False

    online = cache.get_provider_total(
        minutes = minutes, 
        windows = win, 
        total = total - 1, 
        val_end = timestamp_end
        )
    
    x_list, price_list, time_list = online.load_val_data(currency)
    timestamp_start = time_list[0]
    timestamp_end = time_list[-1]
    start = pd.to_datetime(timestamp_start, unit='s')
    end = pd.to_datetime(timestamp_end, unit='s')
    total = len(price_list)
    for idx in range(total):
        price = price_list[idx]
        time = time_list[idx]
        order = [[f"{price}", f"{price}"]]
        amount = 0.0
        agent.process_data(price, amount, time, order, order)
    back.on_down(back.buy_price, back.buy_price)
    if (verbose):
        print(f"###### Past report({total}): {start}({timestamp_start}) - {end}({timestamp_end}) ######")
        back.report()
        print(f"###### {agent.last_timestamp} ######")

def eval_model(
    model, 
    currency, 
    step,
    verbose, 
    provider: OnLineDataProvider,
    cache: CacheProvider,
    hot_load_total):

    valX, valY, time_list = provider.load_val_data(currency)
    
    x_list, y, price_list = get_sequencial_data(valX, valY, step)

    minutes = provider.minutes
    win = provider.windows()
    agent, back, model_agent = get_agent(minutes = minutes,
                                    win = provider.windows(),
                                    step = step,
                                    timestamp = time_list[-1],
                                    currency = currency,
                                    hot_load = False,
                                    verbose = verbose,
                                    model = model,
                                    simulate_on_price = True)
    agent.save_history = True

    #print("aaaa")
    agent_cache_key = f"{minutes}_{win}"
    if agent_cache_key in cache.agent_cache:
        #print(f"Exists {agent_cache_key}")
        cache_list, cache_data = cache.agent_cache[agent_cache_key]
        agent.list = cache_list.copy()
        agent.tec.data = cache_data.copy()
    else:
        #print(f"Creating cache for {agent_cache_key}")
        add_hot_load(
            minutes = minutes, 
            win = win, 
            total = hot_load_total, 
            currency = currency, 
            timestamp_end = time_list[0], #+ (60 * minutes), 
            verbose = verbose, 
            back = back, 
            model_agent = model_agent, 
            agent = agent,
            cache = cache
        )
        cache.agent_cache[agent_cache_key] = (agent.list.copy(), agent.tec.data.copy())
    #print(f"bbbb {agent.tec.data.copy()[-1]}")
    #print(model_agent.history[-1])

    model_agent.verbose = verbose
    back.verbose = verbose

    agent.history = []

    hard_limit = 2000
    if (len(price_list) > hard_limit):
        limit = hard_limit
    else:
        limit = len(price_list)

    for idx in range(limit):
        #if ((idx) != len(agent.history)):
        #    print(f"{idx}) {len(agent.history)} - {len(price_list)}")
        #    20/0
        price = price_list[idx]
        time = time_list[idx]
        order = [[f"{price}", f"{price}"]]
        amount = 0.0
        #print(f"{idx} - {time} - {price} ")
        agent.process_data(price, amount, time, order, order)

    back.on_down(back.buy_price, back.buy_price)
    
    preds = []
    histoty_times = []
    for data in agent.history:
        pred = 0
        if (data.is_up):
            pred = 1
        histoty_times.append(data.timestamp)
        preds.append(pred)

    yy = y[:limit]

    counter = Counter(preds)

    metrics = {}
    if (len(counter) > 1):
        metrics["recall"] = recall_score(yy, preds)
        metrics["precision"] = precision_score(yy, preds, zero_division=1)
        metrics["f1"] = f1_score(yy, preds)
        metrics["accuracy"] = accuracy_score(yy, preds)
        metrics["roc_auc"] = roc_auc_score(yy, preds)

    return back, metrics    


def get_agent(minutes, 
                win, 
                step,
                model,
                currency,
                simulate_on_price,
                hot_load, 
                timestamp,
                verbose
                ):
    
    back = BackTest(value = 100,
                        verbose = verbose,
                        pending_sell_steps = step, 
                        sell_on_profit = True)

    request_sell = lambda bid, ask: back.on_down(bid = bid, ask = ask)
    request_buy = lambda bid, ask: back.on_up(bid = bid, ask = ask)

    model_agent = ModelAgent(
        model = model,
        on_down = request_sell,
        on_up = request_buy,
        verbose = verbose
    )

    model_agent.simulate_on_price = simulate_on_price

    on_new_data = lambda x: print(x)
    on_new_data = lambda x: model_agent.on_x(x)

    on_state = lambda timestamp, price, buy, sell: print("{} {} {} {}".format(timestamp, price, buy, sell))
    on_state = lambda timestamp, price, buy, sell: model_agent.on_new_state(timestamp, price, buy, sell)

    agent = DataAgent(
        tec = TecAn(windows = win, windows_limit = 100),
        minutes = minutes,
        on_state = on_state,
        on_new_data = on_new_data,
        verbose = False
    )
    
    if (hot_load):
        add_hot_load(minutes, 
            win = win, 
            total = 200, 
            currency = currency, 
            timestamp_end = timestamp, #- (60 * minutes), 
            verbose = verbose, 
            back = back, 
            model_agent = model_agent, 
            agent = agent,
            cache = CacheProvider(currency_list = [currency], verbose=False)
        )
        back.verbose = True
    back.reset()
    return agent, back, model_agent
