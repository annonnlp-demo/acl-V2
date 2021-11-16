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

        test_dataset = get_dataset(dataset_root_path + dataset_name + '/' + value + '/', split='test')

        dic[value] = language_(test_dataset
                               , text_name)

    results = {}
    results[dataset_name] = dic

    json_root = './results/'
    Path(json_root).mkdir(parents=True, exist_ok=True)
    json_path = json_root + dataset_name + '-language.json'
    save_json(results, json_path)


def language_(test_dataset,
              text_name):
    import fasttext

    model = fasttext.load_model('colabration/supplement_experiment_by_gram/feature/count_language/lid.176.bin')

    _number = 0
    _other = 0

    for dataset in [test_dataset, ]:
        for value in dataset[text_name]:

            _number = _number + 1

            label, _ = model.predict(value, k=1)
            if label[0] != '__label__en':
                print(value)
                _other = _other + 1
    return (_other / _number)


language('fdu', 'content')
