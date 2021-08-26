from tec_an import *
from data_util import *
from sklearn_model_hyper import *
from data_generator import *
from data_agent import *
from stock_agent import *
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


def order(e):
    return e['profit']
    #return e['result']['profit']

def load_results_path(results_path):
    try:
        saved_models = load(results_path)
        print(f"{results_path} - {len(saved_models)}")
    except:
        saved_models = []
    return saved_models

def load_results_from_path_list(path_list):
    all_models = []

    for path in path_list:
        results = load_results_path(path)
        for result in results:
            if(result.profit >= 100):
                all_models.append(result)
        
    print(f"Pre selected: {len(all_models)}")
    return all_models


def get_scorecoard(
    currency_list, 
    all_models, 
    minutes_list, 
    cache :CacheProvider,
    time_start,
    time_end):

    scoreboard = []

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

        online = cache.get_provider(
            minutes = minutes,
            windows = windows,
            val_start = time_start,
            val_end = time_end
        )
        
        profits = []
        backs = {}
        score = {}
        
        has_negative = False
        
        for currency in currency_list:

            back, metrics = eval_model(
                model=model_detail.model,
                currency=currency,
                step=step,
                provider=online,
                verbose=False,
                cache = cache
            )
            back_profit = back.get_profit()
            profits.append(back_profit)
            backs[currency] = back
            #print(f"{train_set} -> {back_profit}")
            if (back_profit <= 0 and not has_negative):
                has_negative = True
            
        score['profit'] = np.average(profits)
        score['backs'] = backs
        score['model_detail'] = model_detail
        scoreboard.append(score)

    return scoreboard

def get_best_model(currency_list, result_paths, timestamp, minutes_list, winner_path):

    all_models = load_results_from_path_list(result_paths)

    cache = CacheProvider(
        currency_list=currency_list,
        verbose = False
    )

    time_start = timestamp - 36000
    time_end = timestamp

    scoreboard = get_scorecoard(
        currency_list = currency_list, 
        all_models = all_models, 
        minutes_list = minutes_list, 
        cache = cache,
        time_start = time_start,
        time_end = time_end
        )

    selected_count = len(scoreboard)

    if (selected_count == 0):
        print(f"No viable models for currency_list: {currency_list}, result_paths: {result_paths}, timestamp: {timestamp}, minutes_list: {minutes_list}")
        return

    print(f"Selected: {selected_count}")

    scoreboard.sort(key=order, reverse = False)

    filtered = []

    for score in scoreboard[-3:]:
        profit = score['profit']
        backs = score['backs']
        result = score['model_detail']
        print(result)
        for key in backs:
            back = backs[key]
            #backs[back].report()
            trades = f"{back.positive_trades} - {back.negative_trades}"
            print(f"{key} -> {back.current} | {trades}")
        
        print()
        filtered.append(profit)
    
    best = scoreboard[-1]

    winner = best['model_detail']
    print(f"Winner for {currency_list} -> {best['profit']}")
    backs = best['backs']
    for back in backs:
        backs[back].report()
    if (winner_path != None):
        dump(winner, winner_path)
    print()
    return winner

def add_arguments_winner(parser):

    minutes = [30, 15, 5, 3]

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
                nargs='+'
                )

    parser.add_argument('--winner_path',
                dest='winner_path',
                help='Define winner_path',
                action="store"
                )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    add_arguments_winner(parser)

    args = parser.parse_args()

    timestamp = int(datetime.timestamp((datetime.now())))

    print(f"minutes: {args.minutes_list}")
    print(f"currency_list: {args.currency_list}")
    print(f"Result_paths: {args.result_paths_list}")
    print(f"winner_path: {args.winner_path}")

    get_best_model(
        minutes_list=args.minutes_list,
        result_paths=args.result_paths_list,
        currency_list=args.currency_list,
        timestamp = timestamp,
        winner_path = args.winner_path
    )

