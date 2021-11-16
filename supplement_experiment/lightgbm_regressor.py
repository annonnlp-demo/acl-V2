# coding: utf-8
import pandas as pd
from sklearn.metrics import mean_squared_error
import os
import lightgbm as lgb
from data import get_data
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    print('Loading data...')
    # load or create your dataset


    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=40,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=5)



    print('Starting predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

    y_test = np.array(y_test).tolist()
    print(target_name)
    print(stats.spearmanr(y_pred, y_test))








train('variance',['right_grammar','basic_word','right_len'])

train('indicator3',['right_grammar','basic_word','right_len'])


