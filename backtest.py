
def backtest_model(model, x, closed_prices, back):
        
    for idx in range(len(x)):
        xx = [x[idx]]
        yy = model.predict(xx)[0]
        price = closed_prices[idx]
        #print(f'{idx} {yy} {price}')
        back.on_state(0, price)
        if(yy == 1):
            back.request_buy(price)
        else:
            back.request_sell(price)
    return back


def backtest_baseline(x, y, closed_prices, step, back):
    
    for idx in range(len(x)):
        yy = y[idx]
        price = closed_prices[idx]
        #print(f'{idx} {yy} {price}')
        back.on_state(0, price)
        #print(yy)
        if(yy == 1):
            back.request_buy(price)
        else:
            back.request_sell(price)
    return back