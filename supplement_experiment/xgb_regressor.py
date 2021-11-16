import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from data import get_data,  feature_name_list_all
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from numpy import absolute
from matplotlib import pyplot
from xgboost import plot_importance
from scipy import stats

import shap

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
import seaborn as sns # for correlation heatmap
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'



def train(target_name='variance', split_list=None):
    if split_list is None:
        split_list = ['basic_word']

    indicator=38
    if target_name=='variance':
        indicator=38
    elif target_name=='average':
        indicator=39
    elif target_name=='indicator3':
        indicator=40

    data=get_data(split_list)
    X, y = data.iloc[:,3:32],data.iloc[:,indicator]

    X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=123)

    model = XGBRegressor(tree_method='gpu_hist', gpu_id=0)
    '''
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    scores = absolute(scores)
    print('Mean MAE: %.20f (%.20f)' % (scores.mean(), scores.std()) )
    '''
    model.fit(X_train,y_train)
    preds = model.predict(X_test)


    from sklearn.metrics import mean_squared_error

    import numpy as np

    print('RMSE:',np.sqrt(mean_squared_error(y_test,preds)))

    from scipy import stats
    y_test = np.array(y_test).tolist()
    print(target_name)
    print(stats.spearmanr(preds, y_test))

    '''
    plot_importance(model)
    pyplot.figure(figsize=(30, 30))
    pyplot.tight_layout()
    pyplot.show()

    model.feature_importances_()
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
    '''





'''
train('variance',['right_grammar','basic_word','right_len'])

train('indicator3',['right_grammar','basic_word','right_len'])
'''

train('variance',['basic_word',])

train('indicator3',['basic_word',])
