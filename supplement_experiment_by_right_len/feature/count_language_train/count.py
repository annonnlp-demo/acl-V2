import sys
sys.path.append('.././util')

from pathlib import Path

from util import *

dataset_root_path = '../.././dataset/results/'


def get_split_list(dataset_name):
    path = dataset_root_path + dataset_name + '/'
    split_path_list = get_split_dataset_name(path)

    return split_path_list


def language(dataset_name, text_name='text'):
    print('---------' + dataset_name + '---------------')
    dic = {}
    split_name_list = get_split_list(dataset_name)

    for value in split_name_list:
        train_dataset = get_dataset(dataset_root_path + dataset_name + '/' + value + '/')

        dic[value] = language_(
                               train_dataset,
                                text_name)

    results = {}
    results[dataset_name] = dic

    json_root = './results/'
    Path(json_root).mkdir(parents=True, exist_ok=True)
    json_path = json_root + dataset_name + '-language.json'
    save_json(results, json_path)


def language_(
              train_dataset,
               text_name):
    import fasttext

    model = fasttext.load_model('colabration/supplement_experiment_by_gram/feature/count_language/lid.176.bin')

    _number = 0
    _other = 0

    for dataset in [train_dataset]:
        for value in dataset[text_name]:

            _number = _number + 1

            label, _ = model.predict(value, k=1)
            if label[0] != '__label__en':
                print(value)
                _other = _other + 1
    return (_other / _number)


language('ade')
language('ag')
language('dbpedia', 'content')
language('imdb')
language('MR')
language('rotten')
language('fdu', 'content')
language('sst1')
language('sst2')
language('subj')
language('yelp')
