import sys

sys.path.append('.././util')

from pathlib import Path

from util import *

dataset_root_path = '../.././dataset/results/'


def get_split_dataset_name(dataset_name):
    path = '../.././dataset/results/' + dataset_name + '/'
    from os import listdir
    from os.path import join
    list = listdir(path)
    return list


def get_split_dataset_gram(dataset_name, split_name, text_name='text'):
    train_dataset = get_dataset(dataset_root_path + dataset_name + '/' + split_name + '/')

    rw_dic = get_grammar_dic(dataset_name)

    length = 0
    other = 0

    from tqdm import tqdm
    for index in tqdm(range(len(train_dataset))):
        value = train_dataset[index][text_name]

        value_list = value.split(' ')
        length += len(value_list)

        matches = rw_dic[str(train_dataset[index]['index_raw'])] * len(value_list)

        other += matches
    return other / length


def get_daatset_gram(dataset_name, text_name='text', total=None, order=None):
    split_list = get_split_dataset_name(dataset_name)

    json_path = './results/'
    from os import listdir
    import os
    if not os.path.exists(json_path):
        json_file_list = []
    else:
        json_file_list = listdir(json_path)

    tem_split_list = [value for value in split_list]

    for value in tem_split_list:
        for val in json_file_list:
            if (value in val) and (dataset_name in val):
                split_list.remove(value)
                break

    if len(split_list) == 0:
        return

    for value in split_list:
        print('------' + str(order) + '----' + dataset_name + '---' + value + '------')
        dic = {}
        results = get_split_dataset_gram(dataset_name, value, text_name)
        dic[value] = results

        tem_results = {dataset_name: dic}

        json_root = './results/'
        Path(json_root).mkdir(parents=True, exist_ok=True)
        json_path = json_root + dataset_name + '-' + value + '-' + str(order) + '-ppl.json'
        save_json(tem_results, json_path)


def get_grammar_dic(dataset_name):
    path = '../../.././transfer_dataset_with_index/results/grammar/' + dataset_name + '/'
    from os import listdir

    list = listdir(path)

    results = {}

    for value in list:
        tem_dic = get_json(path + value)['grammar']
        results.update(tem_dic)


    return results


'''
dataset_name_list = ['ade', 'ag',  'dbpedia', 'imdb', 'MR',  'rotten',  'fdu', 'sst1', 'sst2',
                     'subj', 'yelp']
dataset_label_number_list = [2, 4,   14, 2, 2,  2,  2, 5, 2, 2, 2]
text_name_list = ['text', 'text',   'content', 'text', 'text',  'text',  'content', 'text',
                  'text', 'text', 'text', ]
'''

get_daatset_gram('dbpedia', 'content')
get_daatset_gram('imdb', )
get_daatset_gram('sst2', )
get_daatset_gram('yelp', )
get_daatset_gram('ag', )

get_daatset_gram('ade')

get_daatset_gram('MR')
get_daatset_gram('rotten')
get_daatset_gram('fdu', 'content')
get_daatset_gram('sst1')

get_daatset_gram('subj')
