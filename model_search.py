
from tec_an import *
from data_util import *
from sklearn_model_hyper import *
from data_generator import *
from data_agent import *
from stock_agent import *
from backtest import *
from bitstamp import *
from model import *


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

from tqdm import tqdm

import argparse

#train_start_list = [1622502000, 1590966000, 1559343600, 1580515200, 1612137600]
train_start_list = [1622502000, 1590966000, 1559343600, 1527807600, 1517443200, 1548979200, 1580515200, 1612137600]
#train_start_list = [1612137600]
train_keys = ["btcusd", "ethusd", "bchbtc"]
#train_keys = ["bchbtc"]
val_start = 1626340500
val_end = 1626369600
load_from_disk = True

# '2Min'

def get_online_data(minutes, source_data_generator, load_from_disk, file_prefix = ""):
    
    online = OnLineDataProvider(
                 source_data_generator = source_data_generator,
                 minutes = minutes,
                 train_keys = train_keys,
                 train_limit = 1000,
                 val_limit = 1000,
                 val_keys = ["btcusd"],
                 val_start = val_start,
                 val_end = val_end,
                 train_start_list = train_start_list
    )

    online_path = f'data/online{file_prefix}_{minutes}'
    
    if (load_from_disk):
        online = load(online_path)    
    else:
        #online.load_train_cache()
        online.load_cache()
        online.sourceDataGenerator = None
        dump(online, online_path)
        
    
    return online

def get_classifiers():
    clss = [
        lambda : CatBoostClassifier(logging_level = 'Silent'),
        
        #lambda : make_pipeline(StandardScaler(),
        #                        MLPClassifier(
        #                            solver='lbfgs', alpha=1, random_state=1, max_iter=20000,
        #                            early_stopping=True)
        #                        ),
        
        lambda : MLPClassifier(alpha=1, random_state=1, max_iter=20000, early_stopping=True),
        
        lambda : xgb.XGBClassifier(random_state=1,
                                   objective = "binary:logistic",
                                   eval_metric='mlogloss',
                                   learning_rate=0.01),
        
        #lambda : GaussianNB(),
        lambda : QuadraticDiscriminantAnalysis(),
        lambda : AdaBoostClassifier(random_state = 42),
        
        lambda : TabNetClassifierEarly(verbose=0),
        
        #lambda : KNeighborsClassifier(3),
        #lambda : GaussianProcessClassifier(1.0 * RBF(1.0)),
        
        
        lambda : DecisionTreeClassifier(random_state = 42),
        #lambda : RandomForestClassifier(max_depth=50, n_estimators=100, max_features=1),
        lambda : rand(),
        
        #lambda : SVC(gamma=2, C=1, random_state = 42),
    ]
    
    def rand():
        ##print("rand created")
        param = {'criterion': 'entropy', 'max_depth': 100, 'n_estimators': 30, 'random_state': 42}
        return RandomForestClassifier(**param)
    

    #clss = []
    #clss.append(lambda : rand())
    #clss.append(lambda : make_pipeline(StandardScaler(),TabNetClassifierEarly(verbose=0)))
    return clss

estimators = [
                lambda : RFE(estimator=RandomForestClassifier(random_state=10), n_features_to_select=10),
                lambda : RFE(estimator=DecisionTreeClassifier(random_state=10), n_features_to_select=8),
                lambda : RFECV(estimator=DecisionTreeClassifier(random_state=10)),
                lambda : RFECV(estimator=RandomForestClassifier(random_state=10))
                ]

estimators = []

#for score in scoring:
    
estimators.append(lambda : RFECV(estimator=DecisionTreeClassifier(random_state=10),
                    scoring='accuracy'))    
estimators.append(lambda : RFECV(estimator=DecisionTreeClassifier(random_state=10),
                    scoring='f1'))    
estimators.append(lambda : RFECV(estimator=DecisionTreeClassifier(random_state=10),
                    scoring='precision'))    

estimators.append(lambda : RFECV(estimator=DecisionTreeClassifier(random_state=10),
                    scoring='recall'))

def add_normalizers(models, cls):
    models.append(make_pipeline(StandardScaler(),cls()))
    models.append(make_pipeline(MinMaxScaler(),cls()))
    models.append(make_pipeline(Normalizer(),cls()))


def get_all_models(indexs_models):
    models = []
    classifiers_temp = get_classifiers()
    
    classifiers = []
    for index in indexs_models:
        classifiers.append(classifiers_temp[index])

    for cls in classifiers:
        models.append(cls())
        add_normalizers(models, cls)
        #models.append(MockModel(cls()))
        #models.append(MockCoPilotModel(cls(), getModel()))
        
        for est in estimators:
            tmp = lambda : Pipeline(
                steps=[
                ('s',est()),
                ('m',cls())
            ])
            add_normalizers(models, tmp)
            #models.append(tmp())
            #models.append(MockModel(tmp()))
            #models.append(MockCoPilotModel(tmp(), getModel()))
    return models


def test_models(provider, get_all_models_factory, steps = [1]):
    
    score_board_by_step = {}

    for step in steps:
        print(f"Step {step}")
        models_list = get_all_models_factory()
        model_result = []
        for idx in tqdm(range(len(models_list))):
            model_creator = models_list[idx]
            model = model_creator
            #print(f"Trainning {model}")
            train_by_step(model, step, provider)
            
            result = run_trial(model, provider, step)

            result['step'] = step

            model_result.append(result)
        
        score_board_by_step[step] = model_result
    
    model_rank = []
    
    def myFunc(e):
        return e['profit']

    for step_idx in score_board_by_step:
        step_board = score_board_by_step[step_idx]
        for result in step_board:
            if (result['profit'] > 101):
                model_rank.append(result)
            
    model_rank.sort(key=myFunc, reverse = True)
    
    return model_rank

def process_model(minu, win, step, model, provider):
        #print(f"Training {model} {provider} ")
        train_by_step(model, step, provider)
        #print(f"Checking {minu} {win} {step} ")
        result = run_trial(model, provider, step)
        result['window'] = win
        result['minutes'] = minu
        result['step'] = step
        #print(result)
        return result

def list():
    models = get_classifiers()
    for i, item in enumerate(models):
        print(f"{i} - {item()}")

def process(minutes, windows, steps, models_index_list):
    
    get_all_models_factory = lambda indexs_models: get_all_models(indexs_models)

    process_list = []

    for minu in minutes:
        for win in windows:
            for step in steps:
                models_list = get_all_models_factory(models_index_list)
                for idx in range(len(models_list)):
                    model_creator = models_list[idx]
                    model = model_creator
                    process = {}
                    process['window'] = win
                    process['minutes'] = minu
                    process['step'] = step
                    process['model'] = model
                    process_list.append(process)
    
    min_results = []
    for process in tqdm(process_list):
        win = process['window'] 
        minu = process['minutes']
        step = process['step'] 
        model = process['model'] 
        
        tec = TecAn(windows = win, windows_limit = 100)
        sourceDataGenerator = SourceDataGenerator(tec = tec)
        provider = get_online_data(minu, sourceDataGenerator, load_from_disk, win)
        
        result = process_model(minu, win, step, model, provider)
        min_results.append(result)

    return min_results

def order_by_proft(e):
    return e['profit']
    #return e['models_profit_metric']['btcusd']

def print_result(result):
    model = result['model']
    window = result['window']
    minutes = result['minutes']
    step = result['step']
    profit = result['profit']
    print(f"Minutes={minutes} Window={window} Step={step} | {profit} {result['models_profit_metric']}")
    print(f"{model}")

def start_model_search(minutes, windows, steps, models_index_list, model_path):
    
    best_results = process(minutes, windows, steps, models_index_list)
    print(f"Results {len(best_results)}")

    #try:
    #    saved_models = load(model_path)
    #except:
    #    saved_models = []
    saved_models = []

    for result in best_results:
        saved_models.append(result)

    print(f"Results saved {len(saved_models)}")
    dump(saved_models, model_path)

    best_results.sort(key=order_by_proft, reverse = False)

    for best in best_results:
        if (best['profit'] < 100):
            continue
        print_result(best)
        #window = best['window']
        #minutes = best['minutes']
        #step = best['step']
        #model_path = f'model/result_{window}_{minutes}_{step}'
        #dump(best, model_path)
        print(f"")

def add_arguments(parser):
    parser.add_argument('-l', '--list',
                            action='store_true',
                            help='List all models'
                            )

    parser.add_argument('-w', '--process_winner',
                        action='store_true',
                        help='search winner'
                        )

    parser.add_argument('--steps',
                    dest='steps',
                    help='Define steps',
                    type=int,
                    default=steps,
                    nargs='+'
                    )

    parser.add_argument('--windows',
                    dest='windows',
                    help='Define windows',
                    type=int,
                    default=windows,
                    nargs='+'
                    )

    parser.add_argument('--minutes',
                    dest='minutes',
                    help='Define minutes',
                    type=int,
                    default=minutes,
                    nargs='+'
                    )

    parser.add_argument('--model_index',
                dest='models_index_list',
                help='Define models indexs',
                type=int,
                default=models_index_list,
                nargs='+'
                )

    parser.add_argument('--model_path',
                dest='model_path',
                help='Define model_path',
                action="store",
                default='model/all_results',
                )

if __name__ == '__main__':

    steps = [1]

    windows = [5, 10, 20, 30, 40]

    minutes = [30, 15, 5, 3]

    models_index_list = [i for i in range(len(get_classifiers()))]

    parser = argparse.ArgumentParser()

    add_arguments(parser)

    args = parser.parse_args()

    if (args.list):
        list()
    else:
        print(f"windows: {args.windows}")
        print(f"minutes: {args.minutes}")
        print(f"steps: {args.steps}")
        print(f"models_index_list: {args.models_index_list}")
        print(f"model_path: {args.model_path}")

        classifiers_temp = get_classifiers()
        
        classifiers = []
        print(f"Processing: ")
        for index in args.models_index_list:
            print(f"Model: {classifiers_temp[index]()}")

        start_model_search(
            steps=args.steps,
            windows=args.windows,
            minutes=args.minutes,
            models_index_list=args.models_index_list,
            model_path=args.model_path,
        )

    
