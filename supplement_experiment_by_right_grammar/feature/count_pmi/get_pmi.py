from datasets import load_from_disk
from datasets import Features, Dataset
import datasets
import string, math, json, os
import nltk


def get_split_dataset_name(dataset_name):
    path = '../.././dataset/results/' + dataset_name + '/'
    from os import listdir
    from os.path import join
    list = listdir(path)

    if 'test' in list:
        list.remove('test')
    if 'train' in list:
        list.remove('train')
    if 'validation' in list:
        list.remove('validation')
    if '__init__.py' in list:
        list.remove('__init__.py')

    return list


def get_train_pmi(dataset_name, label_number, text_name='text', ):
    split_name_list = get_split_dataset_name(dataset_name)
    for value in split_name_list:
        print('-------'+dataset_name+'-------'+value)
        get_train_pmi_(dataset_name, value, label_number, text_name=text_name, )


def get_train_pmi_(dataset_name, split_name, label_number, text_name='text', ):
    dataset_name = dataset_name
    json_file_name = './results/' + dataset_name + '/' + split_name + '/'

    import pathlib
    pathlib.Path(json_file_name).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(json_file_name):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(json_file_name)

    label_column_name = 'label'
    text_name = text_name
    _train_dataset = load_from_disk('../.././dataset/results/' + dataset_name + '/' + split_name + '/train/')
    print(_train_dataset.features)
    punctuation_string = string.punctuation

    label_number = label_number
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
        word_list = _train_dataset.features[label_column_name].names

    for index in range(label_number):
        if '/' in word_list[index]:
            word_list[index] = word_list[index].replace('/', '-')

    train_list_list = []
    total_voca = {}
    total_number = 0

    for index in range(label_number):
        train_list_list.append({'label': word_list[index], 'voca': {}, 'number': 0})
    for index in range(len(_train_dataset)):
        index_ = _train_dataset[index][label_column_name]
        raw_text = _train_dataset[index][text_name]
        raw_text.strip()
        for i in punctuation_string:
            if i == '\'':
                continue
            raw_text = raw_text.replace(i, '')
        tokens = nltk.word_tokenize(raw_text)

        for value in tokens:
            total_number += 1
            train_list_list[index_]['number'] += 1
            if value not in total_voca.keys():
                total_voca[value] = 1
            else:
                total_voca[value] += 1

            if value not in train_list_list[index_]['voca'].keys():
                train_list_list[index_]['voca'][value] = 1
            else:
                train_list_list[index_]['voca'][value] += 1

    pmi_list_list = []
    for index in range(label_number):
        pmi_list_list.append({})

    for index in range(label_number):
        voca = train_list_list[index]['voca']
        words = voca.keys()
        class_number = train_list_list[index]['number']
        for value in words:
            word_total = total_voca[value]
            word_class_number = train_list_list[index]['voca'][value]
            pmi = math.log(
                (word_class_number / total_number) / ((word_total / total_number) * (class_number / total_number)), 2)
            pmi_list_list[index][value] = max(pmi, 0)
        pmi_list_list[index] = sorted(pmi_list_list[index].items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    for index in range(label_number):
        label = word_list[index]
        json_str = json.dumps(pmi_list_list[index], indent=1)
        with open(json_file_name + label + '.json', 'w') as json_file:
            json_file.write(json_str)


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


def get_average_pmi(dataset_name, label_number, text_name='text', ):
    get_train_pmi(dataset_name, label_number, text_name)

    split_name_list = get_split_dataset_name(dataset_name)
    dic = {}

    for value in split_name_list:
        dic[value] = get_average_pmi_(dataset_name, value, label_number, text_name='text', )

    results = {}
    results[dataset_name] = dic

    import json
    json_str = json.dumps(results, indent=1)
    json_path = './results/' + dataset_name + '-pmi.json'
    with open(json_path, 'w') as json_file:
        json_file.write(json_str)


def get_average_pmi_(dataset_name, split_name, label_number, text_name='text', ):
    dataset_name = dataset_name

    label_column_name = 'label'

    test_dataset = load_from_disk('../.././dataset/results/' + dataset_name + '/' + split_name + '/test/')

    punctuation_string = string.punctuation

    label_number = label_number
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
        word_list = test_dataset.features[label_column_name].names
    for index in range(label_number):
        if '/' in word_list[index]:
            word_list[index] = word_list[index].replace('/', '-')

    train_voca_list = []
    for index in range(label_number):
        label = word_list[index]
        j = open('./results/' + dataset_name + '/' + split_name + '/' + label + '.json', 'r')
        python_list = json.load(j)
        train_voca_list.append(python_list)

    total = 0
    result = 0

    for index in range(label_number):
        tem = train_voca_list[index]

        for value in tem:
            total += 1
            result += value[1]

    return result / total
