import os
import string
from glob import glob

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit

import rampwf as rw
from rampwf.score_types.base import BaseScoreType

problem_title = "Isotopic inventory of a nuclear reactor core in operation"

_target_names = [
    j + str(i + 1) for j in list(string.ascii_uppercase) for i in range(80)
]

Predictions = rw.prediction_types.make_regression(label_names=_target_names)
workflow = rw.workflows.Regressor()


class MAPE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="MAPE", precision=4):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        mape = (np.abs(y_true - y_pred) / y_true).mean()
        return mape


score_types = [
    MAPE(name="MAPE"),
]


def get_file_list_from_dir(*, path, datadir):
    data_files = sorted(glob(os.path.join(path, "data", datadir, "*.csv.gz")))
    return data_files


def _get_data(path=".", split="train"):
    

    
    return X, Y


def get_train_data(path="."):
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
    return cv.split(X, y)
