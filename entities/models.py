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
        self.metrics = {}

    def __str__(self):
        return f"TrainedModel (profit = {self.profit}, profit_per_currency = {self.profit_per_currency}, model_detail = {self.model_detail}, metrics = {self.metrics})"



def print_trained_model(trained_model :TrainedModel):
    print(f"{trained_model.profit}")
    for metric_key in trained_model.metrics:
        print(f"{metric_key} -> {trained_model.metrics[metric_key]}")
    print_model_detail(trained_model.model_detail)