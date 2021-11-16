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
    json_path = json_root + dataset_name + '-train-flesh_reading_ease.json'
    save_json(results, json_path)


def ttr_(train_dataset,  text_name):


    results_list=[]

    import textstat

    for value in train_dataset:
        text=value[text_name]
        results=textstat.flesch_reading_ease((text))
        print(results)
        results_list.append(results)



    return sum(results_list)/len(results_list)





    return results


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
