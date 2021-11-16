from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot
# define dataset
from data import get_data
from sklearn.model_selection import train_test_split
# define dataset
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
import lightgbm
import pandas as pd


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
    model = LGBMRegressor()
    '''
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    '''
    # fit the model on the whole dataset
    model = LGBMRegressor()
    model.fit(X_train, y_train)
    # make a single prediction

    yhat = model.predict(X_test)

    from sklearn.metrics import mean_squared_error

    import numpy as np

    print('RMSE:', np.sqrt(mean_squared_error(y_test, yhat)))

    y_test = np.array(y_test).tolist()
    print(target_name)
    print(stats.spearmanr(yhat, y_test))

    #ax = lightgbm.plot_tree(model, tree_index=3, figsize=(20, 8), show_info=['split_gain'])
    #plt.show()

    '''
    booster = model.booster_
    importance = booster.feature_importance(importance_type='split')
    feature_name = booster.feature_name()
    feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance} )
    path='./figure/lgb/'
    from pathlib import Path
    Path(path).mkdir(parents=True,exist_ok=True)

    feature_importance.to_csv(path+target_name+'-feature_importance.csv',index=False)
    '''


'''
train('variance',['basic_word'])

train('indicator3',['basic_word'])


train('variance',['right_len'])

train('indicator3',['right_len'])


train('variance',['right_grammar'])

train('indicator3',['right_grammar'])




'''
train('variance',['right_grammar','basic_word','right_len'])

train('indicator3', ['right_grammar', 'basic_word', 'right_len'])
