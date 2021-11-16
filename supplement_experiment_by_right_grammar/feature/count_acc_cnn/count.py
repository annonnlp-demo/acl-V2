
from pathlib import Path

from supplement_experiment_by_gram.feature.util.util import *
dataset_root_path='../.././model/cnn/results/'

def get_split_list(dataset_name):
    path=dataset_root_path+dataset_name+'/'
    split_path_list=get_split_dataset_name(path)

    return split_path_list

def count_accuracy(dataset_name_list, text_name='text'):

    dic={}

    for value in dataset_name_list:
        tem=count_accuracy_(value)

        for k,v in tem.items():
            dic[k]=v

    json_root='./results/'
    Path(json_root).mkdir(parents=True,exist_ok=True)
    json_path=json_root+'cnn-acc.json'
    save_json(dic,json_path)

def count_accuracy_(dataset_name):
    dic={}
    split_name_list=get_split_list(dataset_name)

    for value in split_name_list:
        tem_dic=get_json(dataset_root_path+dataset_name+'/'+value+'/results/results.json')
        print(tem_dic)
        if tem_dic==None:
            dic[value]=None
        else:
            dic[value]=tem_dic['results']['AccMetric']['acc']

    results={}
    results[dataset_name]=dic

    return results

list=['ade','ag','dbpedia','imdb','MR','rotten','fdu','sst1','sst2','subj','yelp']
count_accuracy(list)





