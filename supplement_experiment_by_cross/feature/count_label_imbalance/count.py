def count_label_percentage(train_dataset,dataset_name):
    if dataset_name=='dbpedia':
        label_number=14
    else:

        label_number = len(train_dataset.features['label'].names)
    number_list = []
    for index in range(label_number):
        number_list.append(0)
    for value in train_dataset:
        label = value['label']
        number_list[label] = number_list[label] + 1

    total_len = len(train_dataset)
    per_list = [value / total_len for value in number_list]
    return per_list, label_number


def count_comentropy(list):
    tem = []
    for value in list:
        if value == 0.0:
            continue
        tem.append(value)

    list = tem
    from math import log
    res = 0
    for value in list:
        print(value)
        res = res - value * log(value, 2)
    return res


def count_count_standard_comentropy(label_size):
    value = 1 / label_size
    from math import log

    res = (0 - value * log(value, 2)) * label_size
    return res


from pathlib import Path

from supplement_experiment_by_gram.feature.util.util import *

dataset_root_path = '../.././dataset/results/'


def get_split_list(dataset_name):
    path = dataset_root_path + dataset_name + '/'
    split_path_list = get_split_dataset_name(path)

    return split_path_list


def label_imbalance(dataset_name, text_name='text'):
    print('---------' + dataset_name + '---------------')
    dic = {}
    split_name_list = get_split_list(dataset_name)

    for value in split_name_list:
        dataset = get_dataset(dataset_root_path + dataset_name + '/' + value + '/')
        dic[value] = label_imbalance_(dataset,dataset_name,text_name)

    results = {}
    results[dataset_name] = dic

    json_root = './results/'
    Path(json_root).mkdir(parents=True, exist_ok=True)
    json_path = json_root + dataset_name + '-label-imbalance.json'
    save_json(results, json_path)


def label_imbalance_(dataset, dataset_name,text_name):
    if dataset_name=='dbpedia':
        return 0
    per_list, label_num = count_label_percentage(dataset,dataset_name)
    comentropy = count_comentropy(per_list)
    standard_comentropy = count_count_standard_comentropy(label_num)

    if standard_comentropy<comentropy:
        print(standard_comentropy)
        print(comentropy)

    return (standard_comentropy - comentropy) / (standard_comentropy)



label_imbalance('fdu', 'content')
