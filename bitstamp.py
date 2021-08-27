import json
import websocket
from tqdm.notebook import tqdm
import requests
import hashlib
import hmac
import time
import requests
import uuid
import datetime
from agents.stock_agent import BackTest
from configparser import ConfigParser
import uuid
from urllib.parse import urlencode

class LiveTrader(BackTest):

    def buy(self, ask):
        super().buy(ask)
 
    def sell(self, sell):
        super().sell(sell)

    def user_transactions(self, limit):
        route = '/api/v2/user_transactions/'
        payload = {
            'limit': str(limit)
        }

        response = _query(self.userinfo, route, payload)
        return response

    def balance(self):
        route = f'/api/v2/balance/'
        response = _query(self.userinfo, route, {})
        return response

    def init(self):
        config_object = ConfigParser()
        config_object.read("config.ini")
        self.userinfo = config_object["bitstamp"]
        self.key = self.userinfo["key"]
        self.secret = self.userinfo["secret"]


def get_now_plus_min(min = 2):
    format = "%a %b %d %H:%M:%S %Y"
    today = datetime.datetime.today() + datetime.timedelta(minutes=min)
    return today.strftime(format)

class LiveBitstamp:
    def __init__(self, list_limit, on_list_full = lambda : print("list full")):
        self.trade = {}
        self.raw_states_list = []
        self.last_timestamp = 0
        self.list_limit = list_limit
        self.last_price = 0
        self.last_time = 0
        self.on_list_full = on_list_full
        self.last_raw_state = {}

    def save_trade(self, data):
        self.trade = data

    #Process the raw state
    def process(self, data):
        current_count = len(self.raw_states_list)
        timestamp = data['timestamp']
        if (timestamp != self.last_timestamp):
            self.raw_states_list.append(data)
            if (current_count < self.list_limit):
                if (current_count == 0):
                  self.pbar = tqdm(total=self.list_limit)
                self.pbar.update(1)
                if (current_count == self.list_limit-1):
                  print("Starting prediction verification at {}".format(get_now_plus_min()))
                  self.pbar.close()
            
        self.last_timestamp = timestamp
        if (len(self.raw_states_list) > self.list_limit):
            self.raw_states_list.pop(0)
            self.on_list_full(self.raw_states_list)
            
    
    def trade_callback(self,data):
        self.last_price = data["price"]
        self.save_trade(data)

    def order_book_callback(self, current_state):
        last_trade = self.trade
        should_process = len(last_trade) > 0
        self.last_time = int(current_state['timestamp'])
        if (should_process):
            orders_limit = 19
            current_state["bids"] = current_state["bids"][:orders_limit]
            current_state["asks"] = current_state["asks"][:orders_limit]
            current_state["amount"] = last_trade["amount"]
            current_state["price"] = last_trade["price"]
            self.last_raw_state = current_state
            self.process(current_state)
    
    def start_live(self):
        bt = Bitstamp(self)
        bt.connect
        self.thread = Thread(target=bt.connect)
        self.thread.daemon = True
        self.thread.start()
        

class Bitstamp:
    def __init__(self, liveStates, currency = "btcusd"):
        self.base_url = "wss://ws.bitstamp.net"
        self.trade_ch = "live_trades_" + currency
        self.bookings_ch = "order_book_" + currency
        self.liveStates = liveStates

    def connect(self):
        #websocket.enableTrace(True)

        self.ws = websocket.WebSocketApp(self.base_url,
                                         on_message=self.__on_message,
                                         on_close=self.__on_close,
                                         on_open=self.__on_open,
                                         on_error=self.__on_error)

        self.ws.run_forever()


    def __on_error(self, ws, error):
        print(str(error))

    def __on_open(self, ws):
        print("Bitstamp Websocket Opened.")
        print("Reading form {} and {}".format(self.trade_ch, self.bookings_ch))
        self.subscribe()

    def __on_close(self, ws):
        print("Bitstamp Websocket Closed.")
        if not self.exited:
            self.reconnect()

    def __on_message(self, ws, message):
        try:
            raw_json = json.loads(message)
            channel = raw_json["channel"]
            data = raw_json["data"]
            if (len(data) != 0):
                #print(channel)
                if (channel == self.bookings_ch):
                    #print("pre")
                    self.liveStates.order_book_callback(data)
                    #print("pos")
                if (channel == self.trade_ch):
                    self.liveStates.trade_callback(data)
        except Exception as e: print("Unexpected error:", e)
        

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
        
def load_bitstamp_ohlc_by_period(currency_pair, start, end, step, verbose = False):
    data = []
    page_start = start
    while (True):
        page = load_bitstamp_ohlc(currency_pair=currency_pair,
                                    start = page_start,
                                    step = step,
                                    limit = 1000,
                                    verbose = verbose
        )
        for item in page:
            data.append(item)
            next_page = int(item['timestamp']) + step
            if (next_page > end):
                return data

        page_start = next_page

        
        
    
def load_bitstamp_ohlc(currency_pair, step, verbose, limit, start, end=-1):
    # params:
    #   currency_pair: currency pair on which to trigger the request
    #   start: unix timestamp from when OHLC data will be started
    #   end: unix timestamp to when OHLC data will be shown
    #   step: timeframe in seconds; possible options are:
    #         60, 180, 300, 900, 1800, 3600, 7200, 14400, 21600, 43200, 86400, 259200
    #   limit: limit ohlc results (minimum: 1; maximum: 1000)
    #
    # returns:
    #   pair: trading pair on which the request was made
    #   high: price high
    #   timestamp: unix timestamp date and time
    #   volume: volume
    #   low: price low
    #   close: closing price
    #   open: opening price

    if step not in [60, 180, 300, 900, 1800, 3600, 7200, 14400, 21600, 43200, 86400, 259200]:
        raise Exception('Invalid step: {}'.format(step))

    if not (1 <= limit and limit <= 1000):
        raise Exception('Invalid limit: {}'.format(limit))

    route = 'https://www.bitstamp.net/api/v2/ohlc/{}/'.format(currency_pair)

    payload = {
        'currency_pair': currency_pair,
        'step': step,
        'limit': limit,
    }
    if (start > 0):
        payload['start'] = start

    if (end > 0):
        payload['end'] = end

    param = ""
    for key in payload:
        value = payload[key]
        param = f"{param}&{key}={value}" 
    
    if(verbose):
        print(f"{route}?{param}")
    
    response = requests.get(route, params=payload)
    ohlc = response.json()
    
    
    return ohlc["data"]["ohlc"]



def _query(info, route, payload):
    key = info['key']
    secret = bytes(info['secret'], 'utf-8')
    nonce = str(uuid.uuid4())
    timestamp = str(int(round(time.time() * 1000)))

    payload_string, content_type = '', ''
    if payload != {}:
        payload_string = urlencode(payload)
        content_type = 'application/x-www-form-urlencoded'

    message = 'BITSTAMP {}POSTwww.bitstamp.net{}{}{}{}v2{}'\
        .format(key, route, content_type, nonce, timestamp, payload_string)\
        .encode('utf-8')

    signature = hmac.new(secret, msg=message, digestmod=hashlib.sha256).hexdigest()

    headers = {
        'X-Auth': f'BITSTAMP {key}',
        'X-Auth-Signature': signature,
        'X-Auth-Nonce': nonce,
        'X-Auth-Timestamp': timestamp,
        'X-Auth-Version': 'v2',
    }

    if content_type != '':
        headers['Content-Type'] = content_type

    full_url = f'https://www.bitstamp.net{route}'

    #print(full_url)

    response = requests.post(full_url,
        headers=headers,
        data=payload_string
    )

    if not response.status_code == 200:
        raise Exception('Status code not 200 on route:{} {}'.format(route, response.status_code))

    if response.reason != 'OK':
        raise Exception('Invalid {} call; reason: {}'.format(route, response.reason))

    content_type = response.headers.get('Content-Type')
    if content_type is None:
        raise Exception('Cannot get Content-Type response header.')

    string_to_sign = (nonce + timestamp + content_type).encode('utf-8') + response.content
    signature_check = hmac.new(secret, msg=string_to_sign, digestmod=hashlib.sha256).hexdigest()

    if not response.headers.get('X-Server-Auth-Signature') == signature_check:
        raise Exception('Signatures do not match')

    content = json.loads(response.content)

    return content

    