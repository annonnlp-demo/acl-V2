from pathlib import Path

import sys
sys.path.append('../../../.././calibration/')
from supplement_experiment_by_gram.feature.util.util import *
from datasets import load_from_disk

dataset_root_path = '../../.././dataset/'


def get_split_list(dataset_name):


    return ['results']


def ttr(dataset_name, text_name='text'):

    print('---------' + dataset_name + '---------------')
    dic = {}
    split_name_list = get_split_list(dataset_name)

    for value in split_name_list:

        test_dataset = load_from_disk(dataset_root_path + dataset_name + '/' + value + '/test/')

        dic[value] = ttr_(test_dataset,  text_name)

    results = {dataset_name: dic}

    json_root = './results/'
    Path(json_root).mkdir(parents=True, exist_ok=True)
    json_path = json_root + dataset_name + '-test-ttr.json'
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


'''
ttr('ade')
ttr('ag')
ttr('dbpedia', 'content')
ttr('imdb')
ttr('MR')
ttr('rotten')
ttr('fdu', 'content')
ttr('sst1')
ttr('sst2')
ttr('subj')
ttr('yelp')
'''


ttr('QC')
ttr('CR')
ttr('atis')
ttr('sms','sms')
