def count_label_percentage(train_dataset,dataset_name):
    if dataset_name=='dbpedia':
        label_number=14
    else:

        label_number = len(train_dataset.features['label'].names)

    return  label_number



from pathlib import Path

from supplement_experiment_by_gram.feature.util.util import *

dataset_root_path = '../.././dataset/results/'


def get_split_list(dataset_name):
    path = dataset_root_path + dataset_name + '/'
    split_path_list = get_split_dataset_name(path)

    return split_path_list


def label_number(dataset_name, text_name='text'):
    print('---------' + dataset_name + '---------------')
    dic = {}
    split_name_list = get_split_list(dataset_name)

    for value in split_name_list:
        dataset = get_dataset(dataset_root_path + dataset_name + '/' + value + '/')
        dic[value] = label_number_(dataset, dataset_name, text_name)

    results = {}
    results[dataset_name] = dic

    json_root = './results/'
    Path(json_root).mkdir(parents=True, exist_ok=True)
    json_path = json_root + dataset_name + '-label-number.json'
    save_json(results, json_path)


def label_number_(dataset, dataset_name, text_name):
    label_num = count_label_percentage(dataset,dataset_name)


    return label_num


label_number('fdu', 'content')
