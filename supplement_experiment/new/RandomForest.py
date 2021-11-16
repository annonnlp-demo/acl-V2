import sys

sys.path.append('../../.././calibration/')
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot
# define dataset
from supplement_experiment.data import get_data
from sklearn.model_selection import train_test_split
# define dataset
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
import lightgbm
import pandas as pd
from sklearn import tree


def RFtrain(X_train, X_test, y_train, y_test,target_name='variance'):

    print('RF')


    # fit the model on the whole dataset
    model = tree.DecisionTreeRegressor()
    model.fit(X_train, y_train)
    # make a single prediction

    yhat = model.predict(X_test)

    from sklearn.metrics import mean_squared_error

    import numpy as np

    print('RMSE:', np.sqrt(mean_squared_error(y_test, yhat)))

    y_test = np.array(y_test).tolist()
    print(target_name)
    print(stats.spearmanr(yhat, y_test))

    cllea,p=stats.spearmanr(yhat, y_test)
    return np.sqrt(mean_squared_error(y_test, yhat)),cllea

