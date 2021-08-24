from entities.entities import *

class TrainedModel():
    
    def __init__(self,
                 model_detail: ModelDetail,
                 profit,
                 profit_per_currency
                 ):
        self.model_detail = model_detail
        self.profit = profit
        self.profit_per_currency = profit_per_currency

    def __str__(self):
        return f"TrainedModel (profit = {self.profit}, model_detail = {self.model_detail})"
