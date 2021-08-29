import pandas as pd
import data_util
from tqdm.notebook import tqdm
#from tqdm import tqdm_notebook as tqdm
from data_generator import DataGenerator
from state_util import StateUtil
from agents.tec_an import TecAn
import numpy as np
from data_util import *
import threading
import multiprocessing
from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
import scikitplot as skplt

from tensorflow.keras import Model, Sequential
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import tensorflow.keras as keras
import random
from catboost import CatBoost
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, classification_report, precision_score, recall_score

def gridSearch(x, y, clf, params, scorer = make_scorer(precision_score)):
    
    clf_grid = GridSearchCV(estimator=clf, 
                            param_grid=params,
                            scoring=scorer
                           )

    clf_grid.fit(x, y)
    best_param = clf_grid.best_params_
    print(best_param)
    print(clf_grid.best_score_)

    return clf_grid


def eval_data(model, x, y):
    pred = np.where(model.predict(x) > 0.5,1,0)
    y_true = y
    y_pred = pred
    target_names = ['class 0', 'class 1']
    print(classification_report(y_true, y_pred, target_names=target_names))
    #print("Confusion Matrix {} {}".format(pred.shape, y.shape))
    #print(skplt.metrics.plot_confusion_matrix(y, pred, normalize=False))
    


def select_features(train_X, train_y, test_X, test_y, model):

    total = train_X.shape[-1]

    feature_names = ['F{}'.format(i) for i in range(total)]


    train_pool = Pool(train_X, train_y, feature_names=feature_names)
    test_pool = Pool(valX, test_y, feature_names=feature_names)

    rfe = RFECV(model,
                #min_features_to_select = 20,
                step=1, 
                verbose=0,
                cv=5)
    rfe = rfe.fit(train_X, train_y)

    print('The mask of selected features: ',rfe.support_)
    print()
    print('The feature ranking:',rfe.ranking_)
    print()
    print('The external estimator:',rfe.estimator_)

    print("Optimal number of features : %d" % rfe.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
    plt.show()

    print(X_reduced.shape)
    return rfe

