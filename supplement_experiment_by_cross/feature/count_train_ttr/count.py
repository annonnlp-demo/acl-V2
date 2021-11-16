from pathlib import Path

import sys
sys.path.append('../../../.././calibration/')

from supplement_experiment_by_pmi.feature.util.util import *


dataset_root_path = '../.././dataset/results/'


def get_split_list(dataset_name):
    path = dataset_root_path + dataset_name + '/'
    split_path_list = get_split_dataset_name(path)

    return split_path_list


def ttr(dataset_name, text_name='text'):

    print('---------' + dataset_name + '---------------')
    dic = {}
    split_name_list = get_split_list(dataset_name)

    for value in split_name_list:
        train_dataset = get_dataset(dataset_root_path + dataset_name + '/' + value + '/')
        test_dataset = get_dataset(dataset_root_path + dataset_name + '/' + value + '/', split='test')
        dev_dataset = get_dataset(dataset_root_path + dataset_name + '/' + value + '/', split='validation')
        dic[value] = ttr_(train_dataset,  text_name)

    results = {dataset_name: dic}

    json_root = './results/'
    Path(json_root).mkdir(parents=True, exist_ok=True)
    json_path = json_root + dataset_name + '-train-ttr.json'
    save_json(results, json_path)


def ttr_(train_dataset,  text_name):


    ttr_list=[]

    from lexicalrichness import LexicalRichness

    for dataset in [train_dataset, ]:
        for value in dataset[text_name]:
            lex = LexicalRichness(value)

            if lex.words==0:
                ttr_list.append(0)
                continue

            ttr_list.append(lex.ttr)





    return sum(ttr_list)/len(ttr_list)


ttr('fdu', 'content')
