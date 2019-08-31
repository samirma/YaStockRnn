import json
import websocket
from time import sleep
from threading import Thread, Lock

trade = None

def save_trade(data):
    global trade
    trade = data
        
def load_trade():
    return trade

      
def save_state(data):
    print(data)
        
    
def trade_callback(data): 
    save_trade(data)

def order_book_callback(current_state):
    last_trade = load_trade()
    if (last_trade is not None):
        orders_limit = 19
        current_state["bids"] = current_state["bids"][:orders_limit]
        current_state["asks"] = current_state["asks"][:orders_limit]
        current_state["amount"] = last_trade["amount"]
        current_state["price"] = last_trade["price"]
        save_state(current_state)
    

class Bitstamp:
    def __init__(self):
        self.base_url = "wss://ws.bitstamp.net"
        self.trade_ch = "live_trades_btcusd"
        self.bookings_ch = "order_book_btcusd"

    def connect(self):
        self.ws = websocket.WebSocketApp(self.base_url,
                                         on_message=self.__on_message,
                                         on_close=self.__on_close,
                                         on_open=self.__on_open,
                                         on_error=self.__on_error)

        self.ws.run_forever()


    def __on_error(self, error):
        print(str(error))

    def __on_open(self):
        print("Bitstamp Websocket Opened.")
        self.subscribe()

    def __on_close(self):
        print("Bitstamp Websocket Closed.")
        if not self.exited:
            self.reconnect()

    def __on_message(self, message):
        raw_json = json.loads(message)
        channel = raw_json["channel"]
        data = raw_json["data"]
        if (channel == self.bookings_ch):
            order_book_callback(data)
        
        if (channel == self.trade_ch):
            print("trade")
            trade_callback(data)

    def isConnected(self):
        return self.ws.sock and self.ws.sock.connected

    def disconnect(self):
        self.exited = True
        self.timer.cancel()
        self.callbacks = {}
        self.ws.close()
        self.logger.info('Bitstamp Websocket Disconnected')

    def reconnect(self):
        if self.ws is not None and self.isConnected():
            self.disconnect()
        self.connect()

    def __restart_ping_timer(self):
        if self.timer.isAlive():
            self.timer.cancel()
        self.timer = Timer(180, self.ping)
        self.timer.start()

    def ping(self):
        self.ws.send('pong')

    def subscribe(self):
        payload = {
            "event": "bts:subscribe",
            "data": {
                "channel": self.trade_ch
            }
        }
        self.ws.send(json.dumps(payload))
        payload = {
            "event": "bts:subscribe",
            "data": {
                "channel": self.bookings_ch
            }
        }
        self.ws.send(json.dumps(payload))
        print("subscribe")