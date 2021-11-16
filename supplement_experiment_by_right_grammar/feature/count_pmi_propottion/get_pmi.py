'''
get_train_pmi('ade',2,)
get_train_pmi('ag',4,)
get_train_pmi('atis',26,)
get_train_pmi('CR',2,)
get_train_pmi('dbpedia',14,'content')
get_train_pmi('imdb',2,)
get_train_pmi('MR',2,)
get_train_pmi('QC',6,)
get_train_pmi('rotten',2,)
get_train_pmi('sms',2,'sms')
get_train_pmi('fdu',2,'content')
get_train_pmi('sst1',5,)
get_train_pmi('sst2',2,)
get_train_pmi('subj',2,)
get_train_pmi('yelp',2,)
'''

from datasets import load_from_disk
from datasets import Features, Dataset
import datasets
import string, math, json, os
import nltk

'''
daset:ade; pmi:0.5045381048344549
daset:ag; pmi:1.1404571308410687
daset:atis; pmi:1.511761156214245
daset:CR; pmi:0.5975448100014463
daset:dbpedia; pmi:2.578798484369395
daset:imdb; pmi:0.5838825562069986
daset:MR; pmi:0.5813443414225481
daset:QC; pmi:1.5560777782361492
daset:rotten; pmi:0.5833003894315555
daset:sms; pmi:0.745260811739424
daset:fdu; pmi:0.5554045620566499
daset:sst1; pmi:1.1022253342690567
daset:sst2; pmi:0.28641858643223395
daset:subj; pmi:0.6366168957129201
daset:yelp; pmi:0.7398695755923451
'''


def get_average_pmi_(dataset_name, split_name, dataset, text_name='text', ):
    label_column_name = 'label'

    if dataset_name=='dbpedia':
        word_list=[
            "Company",
            "EducationalInstitution",
            "Artist",
            "Athlete",
            "OfficeHolder",
            "MeanOfTransportation",
            "Building",
            "NaturalPlace",
            "Village",
            "Animal",
            "Plant",
            "Album",
            "Film",
            "WrittenWork",
        ]
    else:
        word_list = dataset.features[label_column_name].names

    label_number = len(word_list)
    for index in range(label_number):
        if '/' in word_list[index]:
            word_list[index] = word_list[index].replace('/', '-')

    train_voca_list = []
    for index in range(label_number):
        label = word_list[index]
        j = open('.././count_pmi/results/' + dataset_name + '/' + split_name + '/' + label + '.json', 'r')
        python_list = json.load(j)









        tem=python_list[0:10]
        tem=[value[0] for value in tem]



        train_voca_list.append(tem)

    total = len(dataset)
    result = 0

    for index in range(label_number):
        tem = train_voca_list[index]
        tem_dataset = dataset.filter(lambda example: example['label'] == index, keep_in_memory=True)

        for value in tem_dataset:
            text = value[text_name]

            for word in tem:
                if word in text:
                    result += 1
                    break

    return result / total
