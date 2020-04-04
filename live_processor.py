import datetime

from state_util import StateUtil

class LiveProcessor:
  def __init__(self, stateUtil = StateUtil()):
    self.predictions = []
    self.stateUtil = stateUtil
    self.TIMESTAMP_KEY = "timestamp"
    self.ASKS_KEY = "asks"
    self.BIDS_KEY = "bids"
    self.PRICE_KEY = "price"
    self.AMOUNT_KEY = "amount"
    
  def get_now(self):
    format = "%a %b %d %H:%M:%S %Y"
    today = datetime.datetime.today()
    return today.strftime(format)
    
  def get_now_plus_min(self, min = 1):
    format = "%a %b %d %H:%M:%S %Y"
    today = datetime.datetime.today() + datetime.timedelta(minutes=min)
    return today.strftime(format)
    
  def check_predictions(self, predictions, timestamp, current_bid):
    pred_timestamp = predictions[0][0]
    bid_predicted = predictions[0][1]
    should_buy = predictions[0][2]
    if (timestamp > pred_timestamp):
      predictions.pop(0)
      
      is_current_bid_higher_than_predict = current_bid >= bid_predicted
      
      if (should_buy):
        asked = predictions[0][3]
        result = "Result: {} ".format((current_bid - asked))
        time_string = "Current tp: {} Pred tp: {} ".format(datetime.datetime.fromtimestamp(timestamp), datetime.datetime.fromtimestamp(pred_timestamp))
        if (is_current_bid_higher_than_predict):
          print("BUY_CORRECT: Current bid {} bid_predicted: {} {}".format(current_bid, bid_predicted, result))
        else:
          print("BUY_FAIL: Current bid {} bid_predicted: {} {}".format(current_bid, bid_predicted, result))
      
      if (not should_buy):
        if (not is_current_bid_higher_than_predict):
          print("SELL CORRECT: Current bid {} bid_predicted: {} Current tp: {} Pred tp: {}".format(current_bid, bid_predicted, datetime.datetime.fromtimestamp(timestamp), datetime.datetime.fromtimestamp(pred_timestamp)))
        else:
          print("SELL FAIL: Current bid {} bid_predicted: {}".format(current_bid, bid_predicted))
      
  
  def get_processed_data(self, raw_states_list):
    processed_data = []
    for raw_state in raw_states_list:
      state = self.stateUtil.get_parse_state(raw_state)
      processed_data.append(state)
    return np.array(processed_data)
  
  def live_predict(self, raw_states_list, get_model_pred):

    final_pred = get_model_pred(np.array([self.get_processed_data(raw_states_list)]))

    raw_state = raw_states_list[-1]

    ask = float(raw_state[self.ASKS_KEY][0][0])

    timestamp = int(raw_state['timestamp'])
    
    future_timestamp = timestamp + self.stateUtil.future

    current_bid = float(raw_state[self.BIDS_KEY][0][0])

    bid_predicted = self.stateUtil.get_bid_goal(ask)
    if (final_pred == 0):
      print('Should by now for {} and sell later for {} at {}'.format(ask, bid_predicted, self.get_now_plus_min()))
      self.predictions.append([future_timestamp, bid_predicted, True, ask])
    else:
      #print('NO Bid order higher than {}'.format(bid_predicted))
      self.predictions.append([future_timestamp, bid_predicted, False, ask])
      
    self.check_predictions(self.predictions, timestamp, current_bid)
      
