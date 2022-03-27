"""
Starting-kit RAMP
problem.py
"""

# Importing the necesarry packages and librairies

import os
import string
from glob import glob

import numpy as np
import pandas as pd
import pylab as pl
import copy as cp
import string
import re

import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt

import rampwf as rw
from rampwf.score_types.base import BaseScoreType


# Problem Title
problem_title = "Salary Prediction of NBA basketball players"


_target_names = ['SALARY']

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


def get_file_list_from_dir(*, path, datadir):
    data_files = sorted(glob(os.path.join(path, "data", datadir, "*.csv.gz")))
    return data_files


def _get_data(path=".", split="train"):
    # load and concatenate data in one dataset
    # ( train data are composed of 690 different
    # simulations of an operating reactor
    # and test data of 230 simulations)
    # returns X (input) and Y (output) arrays
    data_files = get_file_list_from_dir(path=path, datadir=split)
    dataset = pd.concat([pd.read_csv(f) for f in data_files])

    # Isotopes are named from A to Z
    alphabet = list(string.ascii_uppercase)

    # At T=0 only isotopes from A to H are != 0.
    # Those are the input composition
    # The input parameter space is composed of those initial
    # compositions + operating parameters p1 to p5
    input_params = alphabet[:8] + ["p1", "p2", "p3", "p4", "p5"]

    data = dataset[alphabet].add_prefix("Y_")
    data["times"] = dataset["times"]
    data = data[data["times"] > 0.0]

    temp = pd.DataFrame(
        np.repeat(dataset.loc[0][input_params].values, 80, axis=0),
        columns=input_params
    ).reset_index(drop=True)
    data = pd.concat([temp, data.reset_index(drop=True)], axis=1)

    # data = shuffle(data, random_state=57)

    X_df = (
        data.groupby(input_params)["A"]
        .apply(list)
        .apply(pd.Series)
        .rename(columns=lambda x: "A" + str(x + 1))
        .reset_index()[input_params]
    )
    Y_df = []
    for i in alphabet:
        Y_df.append(
            data.groupby(input_params)["Y_" + i]
            .apply(list)
            .apply(pd.Series)
            .rename(columns=lambda x: i + str(x + 1))
            .reset_index()
            .iloc[:, len(input_params):]
        )
    Y_df = pd.concat(Y_df, axis=1)

    X = X_df.to_numpy()
    Y = Y_df.to_numpy()
    return X, Y


def get_train_data(path="."):
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
    return cv.split(X, y)


