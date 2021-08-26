
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
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier

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
from sklearn.preprocessing import RobustScaler, StandardScaler
from tpot.builtins import StackingEstimator

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
                 train_start_list = train_start_list,
                 verbose = True
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

        lambda : ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.9500000000000001, min_samples_leaf=10, min_samples_split=3, n_estimators=100),
        
        lambda : MLPClassifier(alpha=1, random_state=1, max_iter=20000, early_stopping=True),
        
        lambda : xgb.XGBClassifier(random_state=42),
        
        #lambda : GaussianNB(),
        lambda : QuadraticDiscriminantAnalysis(),
        #lambda : AdaBoostClassifier(random_state = 42),
        
        lambda : TabNetClassifierEarly(verbose=0),
        
        lambda : make_pipeline(
                        RobustScaler(),
                        StandardScaler(),
                        StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.9500000000000001, min_samples_leaf=4, min_samples_split=4, n_estimators=100)),
                        KNeighborsClassifier(n_neighbors=56, p=2, weights="distance")
                    ),
        
        lambda : DecisionTreeClassifier(random_state = 42),
        #lambda : RandomForestClassifier(max_depth=50, n_estimators=100, max_features=1),
        lambda : rand(),
        
        #lambda : SVC(gamma=2, C=1, random_state = 42),
    ]
    
    def rand():
        return RandomForestClassifier(bootstrap=False, 
                                        criterion="entropy",
                                        random_state = 42,
                                        max_features=0.1, 
                                        min_samples_leaf=1, 
                                        min_samples_split=2, 
                                        n_estimators=100)
    

    #clss = []
    #clss.append(lambda : rand())
    #clss.append(lambda : make_pipeline(StandardScaler(),TabNetClassifierEarly(verbose=0)))
    return clss

estimators = []

def rfe_estimator():
    return ExtraTreesClassifier(criterion="gini", max_features=0.3, n_estimators=100, random_state = 42)
                 
estimators.append(lambda : RFE(estimator=rfe_estimator(), step=0.7500000000000001))

estimators.append(lambda : RFECV(estimator=rfe_estimator(), scoring='recall'))

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



def run_trial(trained_model: TrainedModel, provider: OnLineDataProvider, step):
           
    reference_profit = {}
    models_profit = {}

    models_profit_metric = {}

    models_score = {}
    
    cache = CacheProvider(currency_list=provider.val_keys, verbose = False)
    
    profits = []
    for train_set in provider.val_keys:
        trainX_raw, trainY_raw, times = provider.load_val_data(train_set)
        x, y, closed_prices = get_sequencial_data(trainX_raw, trainY_raw, step)
        reference = get_max_profit(x, y, closed_prices, step)

        key = train_set

        reference_profit[key] = reference.get_profit()
        #print(reference.current)

        back, score = eval_model(
                trained_model.model_detail.model, 
                train_set, 
                step = step,
                cache = cache,
                provider = provider,
                hot_load_total = 100,
                verbose = False
                )
        
        #models_profit[key] = f"{back.get_profit()}"
        models_profit[key] = back.get_profit()
        models_score[key] = score
        models_profit_metric[key] = back.get_profit() / reference.get_profit()

        profits.append(back.current)

    trained_model.profit=np.average(profits)
    trained_model.profit_per_currency=models_profit
    return trained_model

def list():
    models = get_classifiers()
    for i, item in enumerate(models):
        print(f"{i} - {item()}")


def train_list(process_list):
    trained_model_list = []
    for process in tqdm(process_list):
        model_detail :ModelDetail = process
        win = model_detail.data_detail.windows
        minu = model_detail.data_detail.minutes
        step = model_detail.data_detail.steps_ahead
        model = model_detail.model
        
        provider = get_provider(minu, win)
        
        train_by_step(model, step, provider)
        
        #result = run_trial(model, provider, step)

        trained_model = TrainedModel(
            profit=100,
            profit_per_currency=None,
            model_detail=model_detail
        )

        trained_model_list.append(trained_model)
    return trained_model_list


def process(minutes_list, windows_list, steps_list, models_index_list, base_path):
    
    get_all_models_factory = lambda indexs_models: get_all_models(indexs_models)

    trained_model_path_list = []

    for minu in minutes_list:
        for win in windows_list:
            process_list = []
            trained_model_path = f"{base_path}_{minu}_{win}"
            print(f"Training {trained_model_path}")
            for steps_ahead in steps_list:
                models_list = get_all_models_factory(models_index_list)
                for idx in range(len(models_list)):
                    model_creator = models_list[idx]
                    model = model_creator
                    data_detail = DataDetail(
                        windows = win,
                        minutes = minu,
                        steps_ahead = steps_ahead,
                    )
                    model_detail = ModelDetail(model=model,data_detail=data_detail)
                    process_list.append(model_detail)
            trained_model_list = train_list(process_list)
            trained_model_path_list.append(trained_model_path)
            dump(trained_model_list, trained_model_path)
    
    return trained_model_path_list
    

def get_provider(minu, win):
    tec = TecAn(windows = win, windows_limit = 100)
    sourceDataGenerator = SourceDataGenerator(tec = tec)
    provider = get_online_data(
            minutes = minu, 
            source_data_generator=sourceDataGenerator, 
            load_from_disk = load_from_disk, 
            file_prefix=win
            )
        
    return provider

def order_by_proft(e:TrainedModel):
    return e.profit
    #return e['models_profit_metric']['btcusd']


def start_model_search(minutes, windows, steps, models_index_list, model_path):
    
    trained_model_path_list = process(
        minutes_list = minutes, 
        windows_list = windows, 
        steps_list = steps, 
        models_index_list = models_index_list,
        base_path = model_path
        )

    for trained_model_path in trained_model_path_list:
        print(f"Scoring {trained_model_path}")
        trained_model_list = load(trained_model_path)
        eval_models(trained_model_list)
        for best in trained_model_list:
            trained:TrainedModel = best
            if (trained.profit < 100):
                continue
            print(trained)
            print(f"")

        trained_model_list.sort(key=order_by_proft, reverse = False)

        dump(trained_model_list, trained_model_path)

def eval_models(models):
    print(f"Recovering profits")
    for model in tqdm(models):
        trained_model :TrainedModel = model
        minutes = trained_model.model_detail.data_detail.minutes
        windows = trained_model.model_detail.data_detail.windows
        steps_ahead = trained_model.model_detail.data_detail.steps_ahead
        provider = get_provider(minu = minutes, win = windows)        
        run_trial(trained_model, provider, steps_ahead)

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

    minutes = [3, 30, 15, 5]

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

    
