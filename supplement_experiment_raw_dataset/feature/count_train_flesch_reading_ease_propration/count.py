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
        train_dataset = load_from_disk(dataset_root_path + dataset_name + '/' + value + '/train/')

        dic[value] = ttr_(train_dataset,  text_name)

    results = {dataset_name: dic}

    json_root = './results/'
    Path(json_root).mkdir(parents=True, exist_ok=True)
    json_path = json_root + dataset_name + '-train-flesh_reading_ease_proportion.json'
    save_json(results, json_path)


def ttr_(train_dataset,  text_name):




    import textstat

    length=len(train_dataset)
    number=0

    for value in train_dataset:
        text=value[text_name]
        results=textstat.flesch_reading_ease((text))

        if results<60:

            number+=1



    return number/length







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