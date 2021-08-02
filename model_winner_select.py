
from tec_an import *
from data_util import *
from sklearn_model_hyper import *
from data_generator import *
from data_agent import *
from stock_agent import *
from backtest import *
from bitstamp import *
from model import *

from model_search import print_result

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import *
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, RFECV
from datetime import datetime

from tqdm import tqdm

import argparse


def order(e):
    return e['profit']
    #return e['models_profit_metric']['btcusd']


def load_online(minutes, window, val_end, currency_list = []):
    tec = TecAn(windows = window, windows_limit = 100)
    source_data_generator = SourceDataGenerator(tec = tec)

    start = val_end - (60 * 100 * minutes)
    end = val_end - (160 * minutes)

    online = OnLineDataProvider(
                 source_data_generator = source_data_generator,
                 minutes = minutes,
                 train_keys = [],
                 train_limit = 40,
                 val_limit = 999,
                 val_keys = currency_list,
                 val_start = start,
                 val_end = end,
                 train_start_list = []
    )

    online.load_cache()
    return online

def test_model(model, set_key, provider, step, verbose = True):
    valX, valY = provider.load_val_data(set_key)

    x, y, closed_prices = get_sequencial_data(valX, valY, step)
    
    #print(len(x))
    preds = model.predict(x)

    recall = recall_score(y, preds)
    precision = precision_score(y, preds)
    f1 = f1_score(y, preds)
    accuracy = accuracy_score(y, preds)

    back = BackTest(value = 100, 
                    verbose = verbose, 
                    sell_on_profit = True,
                    pending_sell_steps = step)
    back = backtest_model(model, x, closed_prices, back)
    return back

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
            if(result['profit'] > 100):
                all_models.append(result)
        
    print(f"Pre selected: {len(all_models)}")
    return all_models


def get_scorecoard(currs, all_models, timestamp, minutes_list):
    scoreboard = []

    online_cache = {}

    for best in tqdm(all_models):
        
        model = best['model']
        window = best['window']
        minutes = best['minutes']
        step = best['step']
        profit = best['profit']

        if minutes not in minutes_list :
            continue

        #print(f"Minutes={minutes} Window={window} Step={step} | {profit}")
        cache_key = f"{minutes}-{window}"
        try:
            online = online_cache[cache_key]
        except :
            #print(f"Not found {cache_key}")
            online = load_online(minutes = minutes, window = window, val_end = timestamp, currency_list = currs)
            online_cache[cache_key] = online
            
        profits = []
        backs = {}
        score = {}
        
        has_negative = False
        
        for train_set in currs:
            back = test_model(model, train_set, online, step, False)
            back_profit = back.get_profit()
            profits.append(back_profit)
            backs[train_set] = back
            #print(f"{train_set} -> {back_profit}")
            if (back_profit <= 0 and not has_negative):
                has_negative = True
            
        score['profit'] = np.average(profits)
        score['backs'] = backs
        score['result'] = best
        if (not has_negative):
            scoreboard.append(score)

    return scoreboard

def get_best_model(currency_list, result_paths, timestamp, minutes_list, winner_path):

    all_models = load_results_from_path_list(result_paths)

    scoreboard = get_scorecoard(currency_list, all_models, timestamp, minutes_list)

    print(f"Selected: {len(scoreboard)}")

    scoreboard.sort(key=order, reverse = True)

    filtered = []

    for score in scoreboard:
        profit = score['profit']
        backs = score['backs']
        #print(f"Current profit: {profit}")
        #print(result)
        #for back in backs:
            #backs[back].report()
        #    print(f"{back} -> {backs[back].get_profit()}")
        
        #print()
        filtered.append(profit)
    
    winner = scoreboard[0]['result']
    print(f"Winner for {currency_list}")
    backs = scoreboard[0]['backs']
    for back in backs:
        backs[back].report()
    print_result(winner)
    if (winner_path != None):
        dump(winner, winner_path)
    print()
    return winner

def add_arguments_winner(parser):

    minutes = [15, 5, 3]

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

