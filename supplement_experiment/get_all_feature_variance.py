from pathlib import Path

import sys

sys.path.append('../../../.././calibration/')

from util import *

feature_part_path_list = [
    '/count_average_length_test/results/',
    '/count_average_length_train/results/',
    '/count_basic_word/results/',
    '/count_basic_word_train/results/',
    '/count_basic_word_test/results/',
    '/count_grammar_test/results/',
    '/count_grammar_train/results/',
    '/count_label_imbalance/results/',
    '/count_label_number/results/',
    '/count_language/results/',
    '/count_language_test/results/',
    '/count_language_train/results/',
    '/count_pmi/results/',
    '/count_pmi_propottion/results/',
    '/count_ppl_test/results/',
    '/count_ppl_train/results/',
    '/count_test_flesch_reading_ease/results/',
    '/count_test_flesch_reading_ease_propration/results/',
    '/count_test_ttr/results/',
    '/count_train_flesch_reading_ease/results/',
    '/count_train_flesch_reading_ease_propration/results/',
    '/count_train_ttr/results/',
]

feature_name_list = [
    'average_length_test',
    'average_length_train',
    'basic_word',
    'basic_word_train',
    'basic_word_test',
    'grammar_test',
    'grammar_train',
    'label_imbalance',
    'label_number',
    'language',
    'language_test',
    'language_train',
    'pmi',
    'pmi_propottion',
    'ppl_test',
    'ppl_train',
    'test_flesch_reading_ease',
    'test_flesch_reading_ease_propration',
    'test_ttr',
    'train_flesch_reading_ease',
    'train_flesch_reading_ease_propration',
    'train_ttr',

]


def get_split_dataset_feature(path):
    from os import listdir

    list = listdir(path)
    return list


def get_all(split_name):
    print(split_name)
    variance_json_file_root = '.././supplement_experiment_by_' + split_name + '/feature'
    variance_json_file_path = '.././supplement_experiment_by_' + split_name + '/feature/count_acc_variance/results/acc-variance.json'
    bert_json_file_path = '.././supplement_experiment_by_' + split_name + '/feature/count_acc_bert/results/bert-acc.json'
    cbow_json_file_path = '.././supplement_experiment_by_' + split_name + '/feature/count_acc_cbow/results/cbow-acc.json'
    cnn_json_file_path = '.././supplement_experiment_by_' + split_name + '/feature/count_acc_cnn/results/cnn-acc.json'
    lstm_json_file_path = '.././supplement_experiment_by_' + split_name + '/feature/count_acc_lstm/results/lstm-acc.json'
    lstm_self_json_file_path = '.././supplement_experiment_by_' + split_name + '/feature/count_acc_lstm_self/results/lstm-self-acc.json'
    feature_path_list = [variance_json_file_root + value for value in feature_part_path_list]
    feature_dic_list = []

    for index in range(len(feature_path_list)):
        tem_results = []
        feature_path = feature_path_list[index]

        split_file_name = get_split_dataset_feature(feature_path)

        for value in split_file_name:
            if not Path(feature_path + value).is_file():
                continue
            tem_dic = get_feature(feature_path + value)
            tem_results.append(tem_dic)

        feature_dic_list.append(tem_results)

    variance_dic = get_variance(variance_json_file_path)
    bert_dic = get_bert_acc(bert_json_file_path)
    cbow_dic = get_cbow_acc(cbow_json_file_path)
    cnn_dic = get_cnn_acc(cnn_json_file_path)
    lstm_dic = get_lstm_acc(lstm_json_file_path)
    lstm_self_dic = get_lstm_self_acc(lstm_self_json_file_path)

    results = {}

    for k, v in variance_dic.items():
        dataset_dic = {}

        for key, value in v.items():
            tem_results = find_all_feature_of_split_dataset(k, key, feature_dic_list)

            if is_contain_empty(tem_results):
                continue

            split_dic = {}

            for index in range(len(feature_name_list)):
                split_dic[feature_name_list[index]] = tem_results[index]

            split_dic['bert_acc'] = bert_dic[k][key]
            split_dic['cbow_acc'] = cbow_dic[k][key]
            split_dic['cnn_acc'] = cnn_dic[k][key]
            split_dic['lstm_acc'] = lstm_dic[k][key]
            split_dic['lstm_self_acc'] = lstm_self_dic[k][key]

            dataset_dic[key] = split_dic

        results[k] = dataset_dic

    json_path = './results/'
    Path(json_path).mkdir(parents=True, exist_ok=True)

    save_json(results, json_path + split_name + '-results.json')


def is_contain_empty(dic):
    for index in range(len(feature_name_list)):
        if dic[index] == None:
            return True

    return False


def get_feature(json_file_path, ):
    return get_json(json_file_path)


def get_variance(variance_json_file_path):
    return get_json(variance_json_file_path)


def get_bert_acc(bert_json_file_path):
    return get_json(bert_json_file_path)


def get_cbow_acc(cbow_json_file_path):
    return get_json(cbow_json_file_path)


def get_cnn_acc(cnn_json_file_path):
    return get_json(cnn_json_file_path)


def get_lstm_acc(lstm_json_file_path):
    return get_json(lstm_json_file_path)


def get_lstm_self_acc(lstm_self_json_file_path):
    return get_json(lstm_self_json_file_path)


def find_all_feature_of_split_dataset(dataset_name, split_dataset_name, feature_list):
    results = []

    for index in range(len(feature_name_list)):
        feature_dic = feature_list[index]
        results.append(find_all_feature_of_split_dataset_(dataset_name, split_dataset_name, feature_dic))

    return results


def find_all_feature_of_split_dataset_(dataset_name, split_dataset_name, feature_dic):
    for val in feature_dic:
        for k, v in val.items():
            if k == dataset_name:
                for keys, value in v.items():
                    if keys == split_dataset_name:
                        return val[k][keys]
            else:
                continue

    return None

'''
split_name = 'right_len'

get_all(split_name)

split_name = 'basic_word'

get_all(split_name)

#split_name = 'cross'

#get_all(split_name)

split_name = 'right_grammar'

get_all(split_name)
'''
split_name = 'cross'

get_all(split_name)