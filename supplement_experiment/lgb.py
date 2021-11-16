import sys
sys.path.append('../../.././calibration/')
from scipy import stats
from lightgbm import LGBMRegressor
from supplement_experiment.util import *
from sklearn.metrics import ndcg_score

def lgbtrain(X_train, X_test, y_train, y_test, target_name='variance'):
    print('lgb')

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

    cllea, p = stats.spearmanr(yhat, y_test)
    return np.sqrt(mean_squared_error(y_test, yhat)), cllea, ndcg_score([y_test], [yhat]),get_ap(y_test, yhat,
                                                                    np.sqrt(mean_squared_error(y_test, yhat)))
