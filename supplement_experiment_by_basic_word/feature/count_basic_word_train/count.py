from pathlib import Path

import sys
sys.path.append('../../../.././calibration/')
from supplement_experiment_by_pmi.feature.util.util import *
from get_1000_basic_words import test

dataset_root_path = '../.././dataset/results/'


def get_split_list(dataset_name):
    path = dataset_root_path + dataset_name + '/'
    split_path_list = get_split_dataset_name(path)

    return split_path_list


def average_basic_word(dataset_name, text_name='text'):
    basic_word_list = test()
    print('---------' + dataset_name + '---------------')
    dic = {}
    split_name_list = get_split_list(dataset_name)

    for value in split_name_list:
        train_dataset = get_dataset(dataset_root_path + dataset_name + '/' + value + '/')

        dic[value] = average_basic_word_(train_dataset,  basic_word_list, text_name)

    results = {dataset_name: dic}

    json_root = './results/'
    Path(json_root).mkdir(parents=True, exist_ok=True)
    json_path = json_root + dataset_name + '-basic-word.json'
    save_json(results, json_path)


def average_basic_word_(train_dataset, basic_word_list, text_name):
    _number = 0
    _other = 0

    for dataset in [train_dataset,]:
        for value in dataset[text_name]:
            value_list = value.split(' ')
            _number = _number + len(value_list)

            for word in value_list:
                lower = word.lower()
                if lower in basic_word_list:
                    _other = _other + 1

    return _other / _number


average_basic_word('ade')
average_basic_word('ag')
average_basic_word('dbpedia', 'content')
average_basic_word('imdb')
average_basic_word('MR')
average_basic_word('rotten')
average_basic_word('fdu', 'content')
average_basic_word('sst1')
average_basic_word('sst2')
average_basic_word('subj')
average_basic_word('yelp')
