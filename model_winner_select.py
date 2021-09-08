from agents.tec_an import *
from data_util import *
from sklearn_model_hyper import *
from data_generator import *
from agents.data_agent import *
from agents.stock_agent import *
from backtest import *
from bitstamp import *
from model import *
from providers import *
from eval_model import *
from entities.models import *

import numpy as np

from joblib import dump, load
from datetime import datetime

from tqdm import tqdm

import argparse

class EvaluetedModel():
    
    def __init__(self,
                model_detail: ModelDetail,
                currency,
                back,
                metrics,
                stop_loss,
                time_start,
                time_end
                 ):
        self.model_detail = model_detail
        self.back:BackTest = back
        self.currency = currency
        self.metrics = metrics
        self.stop_loss = stop_loss
        self.time_start = time_start
        self.time_end = time_end

def print_evalueted_model(score: EvaluetedModel):
    back = score.back
    trades = f"{len(back.positive_trades)} - {len(back.negative_trades)}"
    result = score.model_detail
    print(f"{score.currency} -> {back.current} | {trades}")
    print(f"stop_loss: {score.stop_loss}")
    print_model_detail(result)
    

def order(e :EvaluetedModel):
    return e.back.get_profit()

def load_results_path(results_path):
    try:
        saved_models = load(results_path)
        print(f"{results_path} - {len(saved_models)}")
    except:
        saved_models = []
    return saved_models

def eval_by_time(currency_list, minutes_list, cache, time_start, time_end, all_models,
            stop_loss):
    evaluated_models = []
    for best in tqdm(all_models):
        trained_model :TrainedModel = best 
        model_detail = trained_model.model_detail
        windows = trained_model.model_detail.data_detail.windows
        minutes = trained_model.model_detail.data_detail.minutes
        step = trained_model.model_detail.data_detail.steps_ahead

        if minutes not in minutes_list :
            continue

        current_time = int(datetime.timestamp((datetime.now())))
        if (current_time < (time_end + (minutes * 60))):
            time_end = (time_end - (minutes * 60))

        online:OnLineDataProvider = cache.get_provider(
                minutes = minutes,
                windows = windows,
                val_start = time_start,
                val_end = time_end
            )
            
        score = {}
            
        for currency in currency_list:
            back, metrics = eval_model(
                    stop_loss = stop_loss,
                    model=model_detail.model,
                    currency=currency,
                    step=step,
                    provider=online,
                    verbose=False,
                    cache=cache,
                    hot_load_total=500
                )
            profit = back.get_profit()
            if (profit > 0):
                evalueted_model = EvaluetedModel(
                    currency = currency,
                    metrics = metrics,
                    back = back,
                    model_detail = model_detail,
                    stop_loss = stop_loss,
                    time_start = time_start, 
                    time_end = time_end
                )
                evaluated_models.append(evalueted_model)

    return evaluated_models


def recover_evalueted_list(minutes_list, all_models):
    evaluated_models = []
    for best in tqdm(all_models):
        trained_model :TrainedModel = best 
        model_detail = trained_model.model_detail
        windows = trained_model.model_detail.data_detail.windows
        minutes = trained_model.model_detail.data_detail.minutes
        step = trained_model.model_detail.data_detail.steps_ahead

        if minutes not in minutes_list :
            continue

        score = {}
        
        profit = trained_model.profit
        if (profit > 0 or profit != 100):
            score['profit'] = profit
            score['backs'] = []
            score['model_detail'] = model_detail
            evaluated_models.append(score)
    return evaluated_models


def get_scorecoard(
    currency_list, 
    result_paths, 
    minutes_list, 
    cache :CacheProvider,
    time_start,
    time_end,
    use_trained_profit,
    stop_loss):

    scoreboard = []

    for path in tqdm(result_paths):
        print(f"Evaluating {path}")
        all_models = load_results_path(path)

        evaluated_models = []
        if (use_trained_profit):
            evaluated_models = recover_evalueted_list(minutes_list, all_models)
        else:
            evaluated_models = eval_by_time(
                currency_list, 
                minutes_list, 
                cache, time_start, 
                time_end, 
                all_models,
                stop_loss)

        
        if (len(evaluated_models) > 0):
            evaluated_models.sort(key=order, reverse = False)
            print_evalueted_model(evaluated_models[-1])
            scoreboard.append(evaluated_models[-1])
            scoreboard.sort(key=order, reverse = False)
            #count = len(scoreboard)
            scoreboard = scoreboard[-5:]
        else:
            print("No suitable model was found")

    return scoreboard


def get_best_model(
    currency_list,
    result_paths,
    start_timestamp,
    end_timestamp,    
    minutes_list,
    winner_path, 
    use_trained_profit,
    stop_loss
    ):

    cache = CacheProvider(
        currency_list=currency_list,
        verbose = False
    )

    time_start = start_timestamp - end_timestamp
    time_end = start_timestamp

    scoreboard = get_scorecoard(
        currency_list = currency_list, 
        result_paths = result_paths, 
        minutes_list = minutes_list, 
        cache = cache,
        time_start = time_start,
        time_end = time_end,
        use_trained_profit = use_trained_profit,
        stop_loss = stop_loss
        )

    selected_count = len(scoreboard)

    if (selected_count == 0):
        print(f"No viable models for currency_list: {currency_list}, result_paths: {result_paths}, timestamp: {timestamp}, minutes_list: {minutes_list}")
        return

    print(f"Selected: {selected_count}")

    scoreboard.sort(key=order, reverse = False)

    filtered = []

    for score in scoreboard[-5:]:
        eval_model :EvaluetedModel = score
        profit = eval_model.back.get_profit()
        print_evalueted_model(eval_model)
        print()
        filtered.append(profit)
    
    best:EvaluetedModel = scoreboard[-1]

    winner = best.model_detail
    print(f"Winner for {currency_list} -> {best.back.get_profit()}")
    print_model_detail(winner)
    backs = best.back
    backs.report()
    if (winner_path != None):
        dump(best, winner_path)
    print()
    return winner

def add_arguments_winner(parser):

    minutes = [30, 15, 5, 3]

    start_timestamp = int(datetime.timestamp((datetime.now())))

    path = "model/"
    files = os.listdir(path)
    models = []
    for file in files:
        models.append(f"{path}{file}")
    

    parser.add_argument('--minutes',
                    dest='minutes_list',
                    help='Define minutes',
                    type=int,
                    default=minutes,
                    nargs='+'
                    )

    parser.add_argument('--result_paths',
                dest='result_paths_list',
                help='Define result_paths',
                action="store",
                default=models,
                nargs='+'
                )

    parser.add_argument('--cl',
                dest='currency_list',
                help='Define currency_list',
                action="store",
                nargs='+',
                default=["btcusd", "ethusd", "ltcusd", "ltcbtc"],
                )

    parser.add_argument('--winner_path',
                dest='winner_path',
                help='Define winner_path',
                action="store"
                )

    parser.add_argument('--start',
                dest='start_timstamp',
                help='Start timestamp',
                action="store",
                default=start_timestamp
                )

    parser.add_argument('--end',
                dest='seconds_back',
                help='seconds before start',
                action="store",
                default=259200,
                )

    parser.add_argument('--use_trained_profit', 
                        dest="use_trained_profit", 
                        default=False, 
                        action=argparse.BooleanOptionalAction)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    add_arguments_winner(parser)

    args = parser.parse_args()

    start_string = f"{pd.to_datetime(args.start_timstamp, unit='s')}({args.start_timstamp})"
    end = args.start_timstamp - args.seconds_back
    end_string = f"{pd.to_datetime(end, unit='s')}({end})"

    print(f"minutes: {args.minutes_list}")
    print(f"currency_list --cl: {args.currency_list}")
    print(f"Result_paths --result_paths: {args.result_paths_list}")
    print(f"winner_path: {args.winner_path}")
    print(f"--start: {start_string}")
    print(f"--end: {end_string}")
    print(f"use_trained_profit --use_trained_profit: {args.use_trained_profit}")

    get_best_model(
        minutes_list=args.minutes_list,
        result_paths=args.result_paths_list,
        currency_list=args.currency_list,
        start_timestamp = args.start_timstamp,
        end_timestamp = args.seconds_back,
        winner_path = args.winner_path,
        use_trained_profit = args.use_trained_profit,
        stop_loss=-1
    )

