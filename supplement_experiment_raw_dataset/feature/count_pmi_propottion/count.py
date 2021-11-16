from get_pmi import *

import sys

sys.path.append('.././util')

from pathlib import Path
from datasets import load_from_disk
import sys
sys.path.append('../../../.././calibration/')
from supplement_experiment_by_gram.feature.util.util import *
from get_pmi import get_average_pmi_

dataset_root_path = '../../.././dataset/'


def get_split_list(dataset_name):


    return ['results']


def pmi_pro(dataset_name, text_name='text'):
    print('---------' + dataset_name + '---------------')
    dic = {}
    split_name_list = get_split_list(dataset_name)

    from tqdm import tqdm

    for value in tqdm(split_name_list):
        train_dataset = load_from_disk(dataset_root_path + dataset_name + '/' + value + '/train/')

        dic[value] = pmi_pro_(dataset_name,value,
                              train_dataset,
                              text_name)

    results = {dataset_name: dic}

    json_root = './results/'
    Path(json_root).mkdir(parents=True, exist_ok=True)
    json_path = json_root + dataset_name + '-pmi-pro.json'
    save_json(results, json_path)


def pmi_pro_(dataset_name,split_name,
             train_dataset, text_name):
    return get_average_pmi_(dataset_name,split_name,train_dataset, text_name=text_name, )


'''
pmi_pro('ade')
pmi_pro('ag')

pmi_pro('dbpedia', 'content')
pmi_pro('imdb')
pmi_pro('MR')
pmi_pro('rotten')
pmi_pro('fdu', 'content')
pmi_pro('sst1')
pmi_pro('sst2')
pmi_pro('subj')
pmi_pro('yelp')
'''

pmi_pro('QC')
pmi_pro('CR')
pmi_pro('atis')
pmi_pro('sms','sms')

