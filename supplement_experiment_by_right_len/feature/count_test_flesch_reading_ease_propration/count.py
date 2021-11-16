from pathlib import Path

import sys
sys.path.append('../../../.././calibration/')
from supplement_experiment_by_gram.feature.util.util import *


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
        dic[value] = ttr_(test_dataset,  text_name)

    results = {dataset_name: dic}

    json_root = './results/'
    Path(json_root).mkdir(parents=True, exist_ok=True)
    json_path = json_root + dataset_name + '-test-flesh_reading_ease_proportion.json'
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





    return results


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
