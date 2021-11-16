import pandas as pd
from sklearn.metrics import ndcg_score
import lightgbm as lgb
import sys
from scipy import stats
from supplement_experiment.data import get_data
from sklearn.model_selection import train_test_split
import sys
from supplement_experiment.util import *
from sklearn.utils import shuffle

sys.path.append('../../.././calibration/')


def get_dataset_ranker(iterate_number, size, split_list):
    data_list = []
    for index in range(1, iterate_number + 1):
        data = get_data(size, split_list, random_state=index)
        data_list.append(data)

    return pd.concat(data_list, keys=[f'x{value}' for value in range(1, iterate_number + 1)])


def test():
    data = get_dataset_ranker(2, 2, ['right_grammar', 'basic_word',
                           'right_len'])
    data_ = data[0:5].copy(deep=True)
    print(data_)
    data_ = shuffle(data_)
    print(data_)


def train_lgb_ranker(boosting='gbdt', target_name='variance', group_size=5, number_of_multiple=5):
    feature_list = [
        'average_length_test',
        'average_length_train',
        'basic_word',
        'basic_word_train',
        'basic_word_test',
        'grammar_train',
        'grammar_test',
        'label_imbalance',
        'label_number',
        'language',
        'language_train',
        'language_test',
        'pmi',
        'pmi_propottion',
        'ppl_train',
        'ppl_test',
        'test_flesch_reading_ease',
        'test_flesch_reading_ease_propration',
        'test_ttr',
        'train_flesch_reading_ease',
        'train_flesch_reading_ease_propration',
        'train_ttr',
        'd_ttr',
        'd_length',
        'd_basic_word',
        'd_grammar',
        'd_language',
        'd_ppl',
        'd_fre',
        'd_fre_po',
    ]
    split_list = ['right_grammar', 'basic_word',
                  'right_len']
    target_name = target_name  # 'indicator3'
    size = 987

    if target_name == 'variance':
        indicator = 38
    elif target_name == 'indicator3':
        indicator = 40
    else:
        indicator = 39

    # data = get_data(size, split_list)
    data = get_dataset_ranker(number_of_multiple, size, split_list)
    data = shuffle(data, random_state =123)
    x_all, y_all = data.iloc[:, 3:33], data.iloc[:, indicator]
    x_all = x_all[feature_list]

    threshold_value = get_the_middle(y_all)

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=123)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.125, random_state=123)

    query_train = [group_size for i in range(int(x_train.shape[0] / group_size))]
    sum_ = sum(query_train)
    if x_train.shape[0] - sum_ > 0:
        query_train.append(x_train.shape[0] - sum_)
    y_train = convert_float_to_int(query_train, y_train)

    query_val = [group_size for i in range(int(x_val.shape[0] / group_size))]
    sum_ = sum(query_val)
    if x_val.shape[0] - sum_ > 0:
        query_val.append(x_val.shape[0] - sum_)
    y_val = convert_float_to_int(query_val, y_val)

    query_test = [group_size for i in range(int(x_test.shape[0] / group_size))]
    sum_ = sum(query_test)
    if x_test.shape[0] - sum_ > 0:
        query_test.append(x_test.shape[0] - sum_)
    y_test_original = convert_list_to_group(query_test, y_test.copy(deep=True))
    y_test = convert_float_to_int(query_test, y_test)

    gbm = lgb.LGBMRanker(objective="lambdarank",
                         metric="ndcg",
                         boosting=boosting
                         )
    gbm.fit(x_train, y_train, group=query_train,
            eval_set=[(x_val, y_val)], eval_group=[query_val],
            eval_at=[group_size], early_stopping_rounds=20, )

    score_list = []
    y_group_list = convert_list_to_group(query_test, y_test)
    x_group_list = convert_list_to_group(query_test, x_test)

    for value in x_group_list:
        score_list.append(list(gbm.predict(value, raw_score=True)))

    ndcg_score_list = []
    for index in range(len(y_group_list)):
        ndcg_score_list.append(ndcg_score(y_true=[y_group_list[index]], y_score=[score_list[index]]))

    print(sum(ndcg_score_list) / len(ndcg_score_list))
    print(map_rank(y_test_original, score_list, threshold_value))


if __name__ == "__main__":
    train_lgb_ranker(boosting='gbdt', target_name='indicator3', group_size=9, number_of_multiple=6)
    train_lgb_ranker(boosting='gbdt', target_name='variance', group_size=9, number_of_multiple=6)

    train_lgb_ranker(boosting='gbdt', target_name='indicator3', group_size=7, number_of_multiple=6)
    train_lgb_ranker(boosting='gbdt', target_name='variance', group_size=7, number_of_multiple=6)

    train_lgb_ranker(boosting='gbdt', target_name='indicator3', group_size=5, number_of_multiple=6)
    train_lgb_ranker(boosting='gbdt', target_name='variance', group_size=5, number_of_multiple=6)

    train_lgb_ranker(boosting='gbdt', target_name='indicator3', group_size=3, number_of_multiple=6)
    train_lgb_ranker(boosting='gbdt', target_name='variance', group_size=3, number_of_multiple=6)