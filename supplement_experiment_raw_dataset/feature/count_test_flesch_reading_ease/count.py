from pathlib import Path

import sys
sys.path.append('../../../.././calibration/')
from supplement_experiment_by_pmi.feature.util.util import *
from datasets import load_from_disk

dataset_root_path = '../../.././dataset/'


def get_split_list(dataset_name):


    return ['results']


def fre(dataset_name, text_name='text'):

    print('---------' + dataset_name + '---------------')
    dic = {}
    split_name_list = get_split_list(dataset_name)

    for value in split_name_list:

        test_dataset = load_from_disk(dataset_root_path + dataset_name + '/' + value + '/test/')

        dic[value] = fre_(test_dataset, text_name)

    results = {dataset_name: dic}

    json_root = './results/'
    Path(json_root).mkdir(parents=True, exist_ok=True)
    json_path = json_root + dataset_name + '-test-flesh_reading_ease.json'
    save_json(results, json_path)


def fre_(train_dataset, text_name):


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
fre('ade')
fre('ag')
fre('dbpedia', 'content')
fre('imdb')
fre('MR')
fre('rotten')
fre('fdu', 'content')
fre('sst1')
fre('sst2')
fre('subj')
fre('yelp')
'''

fre('QC')
fre('CR')
fre('atis')
fre('sms','sms')
