import sys

sys.path.append('.././util')

from pathlib import Path

from util import *


dataset_root_path = '../.././dataset/results/'

def get_split_dataset_name(dataset_name):
    path = '../.././dataset/results/' + dataset_name + '/'
    from os import listdir
    from os.path import join
    list = listdir(path)
    return list


def get_split_dataset_gram(dataset_name, split_name, text_name='text'):

    train_dataset = get_dataset(dataset_root_path + dataset_name + '/' + split_name + '/')

    is_bad_rule = lambda rule: rule.message == 'Possible spelling mistake found.' and len(rule.replacements) and \
                               rule.replacements[0][0].isupper()
    import language_tool_python

    tool = language_tool_python.LanguageTool('en-US')

    length = 0
    other = 0

    from tqdm import tqdm
    for index in tqdm(range(len(train_dataset))):
        value = train_dataset[index][text_name]

        value_list = value.split(' ')
        length += len(value_list)
        matches = tool.check(value)
        matches = [rule for rule in matches if not is_bad_rule(rule)]

        other += len(matches)
    return other / length


def get_daatset_gram(dataset_name, text_name='text',total=None,order=None):
    split_list = get_split_dataset_name(dataset_name)

    if total != None and order != None:

        length = len(split_list)

        span = length // total
        start = (order - 1) * span
        end = start + span
        if total == order:
            end = length

        split_list = split_list[start:end]

    import os
    tem_list=os.listdir('./results/')

    def is_contain(tem_list,value):
        for val in tem_list:
            if value in val:
                return True
        return False

    tem=[value for value in split_list]

    for value in tem:
        if is_contain(tem_list,value):
            split_list.remove(value)
            print(value)








    for value in split_list:

        print('------'+str(order)+'----'+dataset_name+'---'+value+'------')
        dic = {}
        results = get_split_dataset_gram(dataset_name, value, text_name)
        dic[value] = results

        tem_results={dataset_name:dic}

        json_root = './results/'
        Path(json_root).mkdir(parents=True, exist_ok=True)
        json_path = json_root + dataset_name +'-'+value+ '-'+str(order)+'-ppl.json'
        save_json(tem_results, json_path)




'''
dataset_name_list = ['ade', 'ag',  'dbpedia', 'imdb', 'MR',  'rotten',  'fdu', 'sst1', 'sst2',
                     'subj', 'yelp']
dataset_label_number_list = [2, 4,   14, 2, 2,  2,  2, 5, 2, 2, 2]
text_name_list = ['text', 'text',   'content', 'text', 'text',  'text',  'content', 'text',
                  'text', 'text', 'text', ]
'''




