import sys

sys.path.append('../../.././calibration/')
from scipy import stats
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.linear_model import LinearRegression


def train(X_train, X_test, y_train, y_test, target_name='variance'):
    x_train_2 = sm.add_constant(X_train)
    est = sm.OLS(y_train, x_train_2)
    est2 = est.fit()
    print(est2.summary())
