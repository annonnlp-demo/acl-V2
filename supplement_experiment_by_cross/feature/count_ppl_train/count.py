import sys

sys.path.append('.././util')

from pathlib import Path

from util import *
from ppl import *

dataset_root_path = '../.././dataset/results/'


def get_split_list(dataset_name):
    path = dataset_root_path + dataset_name + '/'
    split_path_list = get_split_dataset_name(path)

    return split_path_list


def ppl(dataset_name, text_name='text', total=None, order=None):
    split_name_list = get_split_list(dataset_name)

    if total != None and order != None:

        length = len(split_name_list)

        span = length // total
        start = (order - 1) * span
        end = start + span
        if total == order:
            end = length

        split_name_list = split_name_list[start:end]

    ppl_order(dataset_name, split_name_list,order, text_name)


def ppl_(
        train_dataset,
        text_name):
    return mean(train_dataset, text_name)


def ppl_order(dataset_name, split_name_list, order,text_name='text'):
    print('---------' + dataset_name + '---------------')
    dic = {}

    for value in split_name_list:
        print('---------' + dataset_name + '--' + value + '---------------')
        train_dataset = get_dataset(dataset_root_path + dataset_name + '/' + value + '/')
        test_dataset = get_dataset(dataset_root_path + dataset_name + '/' + value + '/', split='test')
        dev_dataset = get_dataset(dataset_root_path + dataset_name + '/' + value + '/', split='validation')
        dic[value] = ppl_(
            train_dataset,
            text_name)

    results = {}
    results[dataset_name] = dic

    json_root = './results/'
    Path(json_root).mkdir(parents=True, exist_ok=True)
    json_path = json_root + dataset_name + '-'+str(order)+'-ppl.json'
    save_json(results, json_path)

