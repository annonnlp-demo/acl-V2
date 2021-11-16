from datasets import load_dataset
from datasets import load_from_disk
from datasets import concatenate_datasets
import os, string, json
import datasets
from util import *

import sys
import sys

sys.path.append('../../../.././calibration/')
from transfer_dataset_with_index.transfer_yelp_dbpedia import get_feature


def get_grammar_dic(dataset_name):
    path = '../.././transfer_dataset_with_index/results/grammar/' + dataset_name + '/'
    from os import listdir

    list = listdir(path)

    results = {}

    for value in list:
        tem_dic = get_json(path + value)['grammar']
        results.update(tem_dic)

    return results


def get_length(text):
    return text.count(' ') + 1


def convert_to_small(dataset, dataset_name, text_name):
    text_list = []
    label_list = []
    index_list = []

    for index in range(len(dataset)):
        item = dataset[index]
        text_list.append(item[text_name])
        label_list.append(item['label'])
        index_list.append(item['index_raw'])

    dic = {}
    dic[text_name] = text_list
    dic['label'] = label_list
    dic['index_raw'] = index_list

    from datasets import Features, Dataset
    features = get_feature(dataset_name)
    new_dataset = Dataset.from_dict(dic, features=features)

    return new_dataset


def split_selected_dataset_to_ttd(sorted_split_dataset_list, text_name):
    train_list = []
    test_list = []
    dev_list = []
    tem_train_list = []

    for index in range(len(sorted_split_dataset_list)):
        tem = sorted_split_dataset_list[index]
        tem = tem.shuffle()
        print(len(tem))
        if ((len(tem) >= 2) & (len(tem) <= 10)):
            tem_dic = tem.train_test_split(test_size=0.5, keep_in_memory=True)
        elif len(tem) == 1:
            tem_dic = {}
            tem_dic['train'] = tem
            tem_dic['test'] = []
        else:
            tem_dic = tem.train_test_split(test_size=0.2, keep_in_memory=True)
        tem_train_list.append(tem_dic['train'])

        if not len(tem_dic['test']) == 0:
            test_list.append(tem_dic['test'])

    for index in range(len(sorted_split_dataset_list)):
        tem = tem_train_list[index]
        if ((len(tem) >= 2) & (len(tem) < 10)):
            tem_dic = tem.train_test_split(test_size=0.5, keep_in_memory=True)
        elif len(tem) == 1:
            tem_dic = {}
            tem_dic['train'] = tem
            tem_dic['test'] = []
        else:
            tem_dic = tem.train_test_split(test_size=0.125, keep_in_memory=True)

        train_list.append(tem_dic['train'])
        if not len(tem_dic['test']) == 0:
            dev_list.append(tem_dic['test'])

    train = concatenate_datasets(train_list)
    test = concatenate_datasets(test_list)
    dev = concatenate_datasets(dev_list)

    return convert_to_small(train, dataset_name, text_name), convert_to_small(dev, dataset_name,
                                                                              text_name), convert_to_small(test,
                                                                                                           dataset_name,
                                                                                                           text_name)


def split_dataset(dataset_name, label_number, size, total, order, split_raw_dataset, sorted_index_list_list):
    dataset_list = select_daatset(total, order, split_raw_dataset, sorted_index_list_list)

    train, dev, test = split_selected_dataset_to_ttd(dataset_list, text_name)

    path = './results/' + dataset_name + '/' + str(len(test) + len(train) + len(dev)) + '-' + str(
        total) + '-' + str(order) + '/'

    import pathlib
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    train.save_to_disk(path + 'train')
    test.save_to_disk(path + 'test/')
    dev.save_to_disk(path + 'validation/')


def select_daatset(total, order, split_raw_dataset, sorted_index_list_list):
    results_list = []

    for i in range(len(split_raw_dataset)):
        tem = select_dataset_(split_raw_dataset[i], sorted_index_list_list[i], total, order)
        results_list.append(tem)

    return results_list


def select_dataset_(single_split_dataset, single_index_list, total, order):
    length = len(single_split_dataset)
    span = length // total
    start = (order - 1) * span
    end = start + span
    if order == total:
        end = length

    index_list = single_index_list[start:end]

    new_dataset = single_split_dataset.select(index_list, keep_in_memory=True)

    return new_dataset


def get_sorted_index(label_number, split_raw_dataset, dataset_name, text_name='text'):
    results = []

    for index in range(label_number):
        tem = get_sorted_index_(split_raw_dataset[index],dataset_name, text_name)
        results.append(tem)

    return results


def get_sorted_index_(s_dataset, dataset_name, text_name='text'):
    rw_dic = get_grammar_dic(dataset_name)

    dic = {}

    from tqdm import tqdm
    for index in tqdm(range(len(s_dataset))):

        dic[index] = rw_dic[str(s_dataset[index]['index_raw'])]

    list = sorted(dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    results = []

    for value in list:
        results.append(value[0])

    return results


def get_raw_dataset(dataset_name):
    train_dataset = load_from_disk('../.././transfer_dataset_with_index/results/' + dataset_name + '/')

    return train_dataset


def split_raw_dataset_by_label(label_number, dataset_):
    print(len(dataset_))
    dataset_label_list = []

    for index in range(label_number):
        tem = dataset_.filter(lambda ex: ex['label'] == index, keep_in_memory=True)
        dataset_label_list.append(tem)

    return dataset_label_list


dataset_name = 'ade'
label_number = 2
text_name = 'text'

raw_dataset = get_raw_dataset(dataset_name)
split_raw_dataset_list = split_raw_dataset_by_label(label_number, raw_dataset)
sorted_index_list = get_sorted_index(label_number, split_raw_dataset_list, dataset_name, text_name=text_name)

length = len(raw_dataset)
size = 5000
total = length // size + 1

for index in range(1, total + 1):
    split_dataset(dataset_name, label_number, size, total, index, split_raw_dataset_list, sorted_index_list)

dataset_name = 'ag'
label_number = 4
text_name = 'text'

raw_dataset = get_raw_dataset(dataset_name)
split_raw_dataset_list = split_raw_dataset_by_label(label_number, raw_dataset)
sorted_index_list = get_sorted_index(label_number, split_raw_dataset_list, dataset_name, text_name=text_name)

length = len(raw_dataset)
size = 5000
total = length // size

for index in range(1, total + 1):
    split_dataset(dataset_name, label_number, size, total, index, split_raw_dataset_list, sorted_index_list)

dataset_name = 'imdb'
label_number = 2
text_name = 'text'

raw_dataset = get_raw_dataset(dataset_name)
split_raw_dataset_list = split_raw_dataset_by_label(label_number, raw_dataset)
sorted_index_list = get_sorted_index(label_number, split_raw_dataset_list, dataset_name, text_name=text_name)

length = len(raw_dataset)
size = 5000
total = length // size

for index in range(1, total + 1):
    split_dataset(dataset_name, label_number, size, total, index, split_raw_dataset_list, sorted_index_list)

dataset_name = 'MR'
label_number = 2
text_name = 'text'

raw_dataset = get_raw_dataset(dataset_name)
split_raw_dataset_list = split_raw_dataset_by_label(label_number, raw_dataset)
sorted_index_list = get_sorted_index(label_number, split_raw_dataset_list, dataset_name, text_name=text_name)

length = len(raw_dataset)
size = 5000
total = length // size

for index in range(1, total + 1):
    split_dataset(dataset_name, label_number, size, total, index, split_raw_dataset_list, sorted_index_list)

dataset_name = 'rotten'
label_number = 2
text_name = 'text'

raw_dataset = get_raw_dataset(dataset_name)
split_raw_dataset_list = split_raw_dataset_by_label(label_number, raw_dataset)
sorted_index_list = get_sorted_index(label_number, split_raw_dataset_list, dataset_name, text_name=text_name)

length = len(raw_dataset)
size = 5000
total = length // size + 1

for index in range(1, total + 1):
    split_dataset(dataset_name, label_number, size, total, index, split_raw_dataset_list, sorted_index_list)

dataset_name = 'fdu'
label_number = 2
text_name = 'content'

raw_dataset = get_raw_dataset(dataset_name)
split_raw_dataset_list = split_raw_dataset_by_label(label_number, raw_dataset)
sorted_index_list = get_sorted_index(label_number, split_raw_dataset_list, dataset_name, text_name=text_name)

length = len(raw_dataset)
size = 5000
total = length // size

for index in range(1, total + 1):
    split_dataset(dataset_name, label_number, size, total, index, split_raw_dataset_list, sorted_index_list)

dataset_name = 'sst1'
label_number = 5
text_name = 'text'

raw_dataset = get_raw_dataset(dataset_name)
split_raw_dataset_list = split_raw_dataset_by_label(label_number, raw_dataset)
sorted_index_list = get_sorted_index(label_number, split_raw_dataset_list, dataset_name, text_name=text_name)

length = len(raw_dataset)
size = 5000
total = length // size

for index in range(1, total + 1):
    split_dataset(dataset_name, label_number, size, total, index, split_raw_dataset_list, sorted_index_list)

dataset_name = 'sst2'
label_number = 2
text_name = 'text'

raw_dataset = get_raw_dataset(dataset_name)
split_raw_dataset_list = split_raw_dataset_by_label(label_number, raw_dataset)
sorted_index_list = get_sorted_index(label_number, split_raw_dataset_list, dataset_name, text_name=text_name)

length = len(raw_dataset)
size = 5000
total = length // size

for index in range(1, total + 1):
    split_dataset(dataset_name, label_number, size, total, index, split_raw_dataset_list, sorted_index_list)

dataset_name = 'subj'
label_number = 2
text_name = 'text'
raw_dataset = get_raw_dataset(dataset_name)
split_raw_dataset_list = split_raw_dataset_by_label(label_number, raw_dataset)
sorted_index_list = get_sorted_index(label_number, split_raw_dataset_list, dataset_name, text_name=text_name)

length = len(raw_dataset)
size = 5000
total = length // size

for index in range(1, total + 1):
    split_dataset(dataset_name, label_number, size, total, index, split_raw_dataset_list, sorted_index_list)

dataset_name = 'dbpedia'
label_number = 14
text_name = 'content'
raw_dataset = get_raw_dataset(dataset_name)
split_raw_dataset_list = split_raw_dataset_by_label(label_number, raw_dataset)
sorted_index_list = get_sorted_index(label_number, split_raw_dataset_list, dataset_name, text_name=text_name)

length = len(raw_dataset)
size = 5000
total = length // size

for index in range(1, total + 1):
    split_dataset(dataset_name, label_number, size, total, index, split_raw_dataset_list, sorted_index_list)

dataset_name = 'yelp'
label_number = 2
text_name = 'text'
raw_dataset = get_raw_dataset(dataset_name)
split_raw_dataset_list = split_raw_dataset_by_label(label_number, raw_dataset)
sorted_index_list = get_sorted_index(label_number, split_raw_dataset_list, dataset_name, text_name=text_name)

length = len(raw_dataset)
size = 5000
total = length // size

for index in range(1, total + 1):
    split_dataset(dataset_name, label_number, size, total, index, split_raw_dataset_list, sorted_index_list)
