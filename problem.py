import os
import string
from glob import glob

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit

import rampwf as rw
from rampwf.score_types.base import BaseScoreType


problem_title = "Salary predictio of NBA Basketball players"

_target_names = [
    'Salary'
]

Predictions = rw.prediction_types.make_regression(label_names=_target_names)
workflow = rw.workflows.Regressor()


class R2(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="r2_score", precision=4):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        return r2


score_types = [
    R2(name="r2_score"),
]


def get_train_data(path="./"):
    train = pd.read_csv(path + '/data/' + 'train.csv')
    features = list(set((train.columns))- set(['SALARY']))
    X_train, y_train = train[features], train['SALARY']
    X_train = X_train.reset_index()
    X_train = X_train.set_index(['index', 'Season'])
    return X_train.values, y_train.values.reshape(-1, 1)


def get_test_data(path="./"):
    test = pd.read_csv(path + '/data/' + 'test.csv')
    features = list(set((test.columns))- set(['SALARY']))
    X_test, y_test = test[features], test['SALARY']
    X_test = X_test.reset_index()
    X_test = X_test.set_index(['index', 'Season'])
    return X_test.values, y_test.values.reshape(-1, 1)


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
    return cv.split(X, y)
