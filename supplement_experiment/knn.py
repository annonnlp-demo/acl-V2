import sys

sys.path.append('../../.././calibration/')
from supplement_experiment.data import get_data
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn import neighbors
from sklearn.metrics import ndcg_score
from supplement_experiment.util import *

neighbors.KNeighborsRegressor()


def knntrain(X_train, X_test, y_train, y_test, target_name='variance'):
    print('knn')

    # fit the model on the whole dataset
    model = neighbors.KNeighborsRegressor()
    model.fit(X_train, y_train)
    # make a single prediction

    yhat = model.predict(X_test)

    from sklearn.metrics import mean_squared_error

    import numpy as np

    print('RMSE:', np.sqrt(mean_squared_error(y_test, yhat)))

    y_test = np.array(y_test).tolist()
    print(target_name)
    print(stats.spearmanr(yhat, y_test))

    cllea, p = stats.spearmanr(yhat, y_test)
    return np.sqrt(mean_squared_error(y_test, yhat)), cllea,ndcg_score([y_test], [yhat]), get_ap(y_test, yhat,
                                                                    np.sqrt(mean_squared_error(y_test, yhat)))


def train(target_name='variance', split_list=None):
    if split_list is None:
        split_list = ['basic_word']

    indicator = 38
    if target_name == 'variance':
        indicator = 38
    elif target_name == 'average':
        indicator = 39
    elif target_name == 'indicator3':
        indicator = 40

    data = get_data(987, split_list)
    X, y = data.iloc[:, 3:33], data.iloc[:, indicator]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    # evaluate the model

    # fit the model on the whole dataset
    model = neighbors.KNeighborsRegressor()
    model.fit(X_train, y_train)
    # make a single prediction

    yhat = model.predict(X_test)

    from sklearn.metrics import mean_squared_error

    import numpy as np

    print('RMSE:', np.sqrt(mean_squared_error(y_test, yhat)))

    y_test = np.array(y_test).tolist()
    print(target_name)
    print(stats.spearmanr(yhat, y_test))


'''train('variance',['right_grammar','basic_word','right_len'])

train('indicator3', ['right_grammar', 'basic_word', 'right_len'])'''
