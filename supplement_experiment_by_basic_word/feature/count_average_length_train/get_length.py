
from pathlib import Path

import sys
sys.path.append('../../../.././calibration/')
from supplement_experiment_by_pmi.feature.util.util import *
dataset_root_path='../.././dataset/results/'

def get_split_list(dataset_name):
    path=dataset_root_path+dataset_name+'/'
    split_path_list=get_split_dataset_name(path)

    return split_path_list

def average_length(dataset_name,text_name='text'):
    print('---------'+dataset_name+'---------------')
    dic={}
    split_name_list=get_split_list(dataset_name)

    for value in split_name_list:
        dataset=get_dataset(dataset_root_path+dataset_name+'/'+value+'/')
        dic[value]=average_length_(dataset,text_name)

    results={}
    results[dataset_name]=dic

    json_root='./results/'
    Path(json_root).mkdir(parents=True,exist_ok=True)
    json_path=json_root+dataset_name+'-average-length.json'
    save_json(results,json_path)

def average_length_(dataset,text_name):
    size = len(dataset)

    total_length = 0
    from tqdm import tqdm
    for value in tqdm(dataset):
        total_length = total_length + value[text_name].count(' ') + 1

    return total_length / size


average_length('ade')
average_length('ag')
average_length('dbpedia','content')
average_length('imdb')
average_length('MR')
average_length('rotten')
average_length('fdu','content')
average_length('sst1')
average_length('sst2')
average_length('subj')
average_length('yelp')





