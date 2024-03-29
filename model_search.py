
from data_util import *
#from sklearn_model_hyper import *
from data_generator import *
from agents.data_agent import *
from agents.stock_agent import *
from agents.tec_an import TecAn
from backtest import *
from bitstamp import *
from model import *
from providers import *
from eval_model import *
from entities.models import *
from sklearn.naive_bayes import BernoulliNB, GaussianNB
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier

from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.pipeline import make_pipeline, make_union
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
from sklearn.decomposition import FastICA
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import SelectPercentile, f_classif
from tpot.builtins import StackingEstimator, ZeroCount
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, RFECV

from tqdm import tqdm

import argparse


load_from_disk = True

# '2Min'

def get_online_data(minutes, source_data_generator, load_from_disk, file_prefix = ""):
    
    online_path = f'data/online{file_prefix}_{minutes}'
        
    return load(online_path)  

def get_classifiers():
    clss = [
        lambda : CatBoostClassifier(logging_level = 'Silent'),
        
        lambda : BernoulliNB(alpha=100.0, fit_prior=False),

        lambda : ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.9500000000000001, min_samples_leaf=10, min_samples_split=3, n_estimators=100),
        
        lambda : MLPClassifier(alpha=1, random_state=1, max_iter=20000, early_stopping=True),
        
        lambda : xgb.XGBClassifier(random_state=42),
        
        #lambda : GaussianNB(),
        lambda : QuadraticDiscriminantAnalysis(),
        #lambda : AdaBoostClassifier(random_state = 42),
        
        lambda : TabNetClassifierEarly(verbose=0),
        
        lambda : make_pipeline(
                                SelectPercentile(score_func=f_classif, percentile=18),
                                FastICA(tol=0.8500000000000001),
                                BernoulliNB(alpha=10.0, fit_prior=False)
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
    models.append(make_pipeline(RobustScaler(),cls()))
    

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
            #continue
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



def run_trial(
    trained_model: TrainedModel, 
    provider: OnLineDataProvider, 
    step,
    stop_loss,
    cache: CacheProvider):
           
    reference_profit = {}
    models_profit = {}

    models_profit_metric = {}

    models_score = {}
    
    profits = []
    for train_set in provider.val_keys:
        trainX_raw, trainY_raw, times = provider.load_val_data(train_set)
        x, y, closed_prices = get_sequencial_data(trainX_raw, trainY_raw, step)
        reference = get_max_profit(x, y, closed_prices, step, stop_loss=stop_loss)

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
                verbose = False,
                stop_loss=stop_loss
                )
        
        #models_profit[key] = f"{back.get_profit()}"
        models_profit[key] = back.get_profit()
        models_score[key] = score
        models_profit_metric[key] = back.get_profit() / reference.get_profit()

        profits.append(back.get_profit())

    trained_model.profit=np.average(profits)
    trained_model.profit_per_currency=models_profit
    trained_model.metrics=models_score
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


def get_train_detail_list(minutes_list, windows_list, base_path):
    train_model_detail_list = []
    for minutes in minutes_list:
        for win in windows_list:
            trained_model_path = f"{base_path}_{minutes}_{win}"
            data = (minutes, win, trained_model_path)
            train_model_detail_list.append(data)
    return train_model_detail_list

def train_detail_list(steps_list, models_index_list, train_model_detail_list):
    
    get_all_models_factory = lambda indexs_models: get_all_models(indexs_models)

    for data in train_model_detail_list:
        minutes, win, trained_model_path = data
        process_list = []
        print(f"Training {trained_model_path}")
        for steps_ahead in steps_list:
            models_list = get_all_models_factory(models_index_list)
            for idx in range(len(models_list)):
                model_creator = models_list[idx]
                model = model_creator
                data_detail = DataDetail(
                    windows = win,
                    minutes = minutes,
                    steps_ahead = steps_ahead,
                )
                model_detail = ModelDetail(model=model,data_detail=data_detail)
                process_list.append(model_detail)
        trained_model_list = train_list(process_list)
        dump(trained_model_list, trained_model_path)
        print(f"Saved {trained_model_path} {len(trained_model_list)}")


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


def start_model_search(
    minutes, 
    windows, 
    steps, 
    models_index_list, 
    model_path,
    stop_loss, 
    just_evaluate_model):

    train_model_detail_list = get_train_detail_list(
        minutes_list = minutes, 
        windows_list = windows,
        base_path = model_path,
    )

    if (not just_evaluate_model):
        train_detail_list(
            steps_list = steps, 
            models_index_list = models_index_list,
            train_model_detail_list = train_model_detail_list
            )

    evaluate_trained_model_List(
        train_model_detail_list = train_model_detail_list,
        stop_loss = stop_loss
        )

def evaluate_trained_model_List(train_model_detail_list, stop_loss):
    for data in train_model_detail_list:
        minutes, win, trained_model_path = data
        print(f"Scoring {trained_model_path}")
        trained_model_list = load(trained_model_path)
        eval_models(trained_model_list, stop_loss)
        suitable_model_list = []
        for best in trained_model_list:
            trained:TrainedModel = best
            if (trained.profit <= 0):
                continue
            suitable_model_list.append(trained)

        suitable_model_list.sort(key=order_by_proft, reverse = False)
        if(len(suitable_model_list) > 0):
            suitable_model_list = suitable_model_list[-5:]
            for best in suitable_model_list:
                trained:TrainedModel = best
                if (trained.profit <= 0):
                    continue
                print(trained.profit)
                print(trained.model_detail.model)
                print(trained.metrics)
                print(f"")

            print("##### Best model #####")
            winner:TrainedModel = suitable_model_list[-1]
            print(f"{winner.profit} -> {winner.model_detail.data_detail}")
            print(winner.model_detail.model)
            print(winner.metrics)
            print(f"{trained_model_path} saved with {len(suitable_model_list)}")
        else:
            print("No suitable model found")
        dump(suitable_model_list, trained_model_path)

def eval_models(models, stop_loss):
    print(f"Recovering profits")
    trained_model :TrainedModel = models[0]
    minutes = trained_model.model_detail.data_detail.minutes
    windows = trained_model.model_detail.data_detail.windows
    steps_ahead = trained_model.model_detail.data_detail.steps_ahead
    provider = get_provider(minu = minutes, win = windows)     
    cache = CacheProvider(currency_list=provider.val_keys, verbose = False)
    for model in tqdm(models):
        trained_model :TrainedModel = model
        minutes = trained_model.model_detail.data_detail.minutes
        windows = trained_model.model_detail.data_detail.windows
        steps_ahead = trained_model.model_detail.data_detail.steps_ahead
        provider = get_provider(minu = minutes, win = windows)        
        run_trial(trained_model, provider, steps_ahead, cache=cache, stop_loss=stop_loss)

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

    parser.add_argument('--just_evaluate_model', dest="just_evaluate_model", default=False, action=argparse.BooleanOptionalAction)


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
        print(f"just_evaluate_model --just_evaluate_model: {args.just_evaluate_model}")

        classifiers_temp = get_classifiers()
        
        classifiers = []
        print(f"Processing: ")
        for index in args.models_index_list:
            print(f"Model: {classifiers_temp[index]()}")


        stop_loss = -1

        start_model_search(
            steps=args.steps,
            windows=args.windows,
            minutes=args.minutes,
            models_index_list=args.models_index_list,
            model_path=args.model_path,
            stop_loss=stop_loss,
            just_evaluate_model = args.just_evaluate_model
        )

    
