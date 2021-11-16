import pandas as pd

names = ['sentence', 'true', 'bert', 'cnn', 'lstm-self', 'lstm']


def count(dataset_name='atis'):
    real_file = './results/' + dataset_name + '.tsv'
    df = pd.read_csv(real_file, sep='\t', names=names)
    length = len(df)

    compare_list = {}
    na_list = ['bert', 'cnn','lstm-self', 'lstm' ]
    compare_name_list = []
    for i in range(len(na_list)):
        for j in range(i + 1, len(na_list)):
            compare_name_list.append(na_list[i] + '>' + na_list[j])
            compare_name_list.append(na_list[j] + '>' + na_list[i])
            compare_list[na_list[i] + '>' + na_list[j]] = 0
            compare_list[na_list[j] + '>' + na_list[i]] = 0

    results = []

    from tqdm import tqdm
    results.append(get_count_results(df,-1))
    for i in tqdm(range(1000)):
        tem_dataframe = df.iloc[get_random_index(length), :]
        tem = get_count_results(tem_dataframe, i)
        results.append(tem)

        for index in range(len(compare_name_list)):
            compare_list[compare_name_list[index]] += tem[str(i)][compare_name_list[index]]

    save_json(results, './comapre/' + dataset_name + '.json')



    facts=[]
    acc_bert=results[0]["acc_bert"]
    acc_cnn=results[0]['acc_cnn']
    acc_lstm_self=results[0]['acc_lstm_self']
    acc_lstm=results[0]['lstm']

    if acc_bert>acc_cnn:
        facts.append('bert>cnn')
    else:
        facts.append('cnn>bert')

    if acc_bert>acc_lstm_self:
        facts.append('bert>lstm-self')
    else:
        facts.append('lstm-self>bert')

    if acc_bert>acc_lstm:
        facts.append('bert>lstm')
    else:
        facts.append('lstm>bert')

    if acc_lstm_self>acc_lstm:
        facts.append('lstm-self>lstm')
    else:
        facts.append('lstm>lstm-self')

    if acc_lstm_self>acc_cnn:
        facts.append('lstm-self>cnn')
    else:
        facts.append('cnn>lstm-self')

    if acc_lstm>acc_cnn:
        facts.append('lstm>cnn')
    else:
        facts.append('cnn>lstm')


    compare_list['fact'] = {
        facts[0]: compare_list[facts[0]] / 1000,
        facts[1]: compare_list[facts[1]] / 1000,
        facts[2]: compare_list[facts[2]] / 1000,
        facts[3]: compare_list[facts[3]] / 1000,
        facts[4]: compare_list[facts[4]] / 1000,
        facts[5]: compare_list[facts[5]] / 1000,

    }
    compare_list['fact']['final'] = (compare_list['fact'][facts[0]] + compare_list['fact'][facts[1]] + \
                                     compare_list['fact'][facts[2]] + compare_list['fact'][facts[3]] + \
                                     compare_list['fact'][facts[4]] + compare_list['fact'][facts[5]]) / 6


    save_json(compare_list, './compare_all/' + dataset_name + '.json')


def save_json(object, path):
    import json
    json_str = json.dumps(object, indent=1)
    with open(path, 'w') as json_file:
        json_file.write(json_str)


def get_random_index(length):
    import random
    list = [i for i in range(0, length)]

    results = random.sample(list, 300)
    return results



def get_count_results(dataframe, order=1):
    true_list = dataframe[names[1]].to_list()
    bert_list = dataframe[names[2]].to_list()
    cnn_list = dataframe[names[3]].to_list()
    lstm_self_list = dataframe[names[4]].to_list()
    lstm_list = dataframe[names[5]].to_list()

    length = len(true_list)

    import numpy as np

    bert_acc = np.sum(np.array(true_list) == np.array(bert_list)) / length
    cnn_acc = np.sum(np.array(true_list) == np.array(cnn_list)) / length

    lstm_acc = np.sum(np.array(true_list) == np.array(lstm_list)) / length
    lstm_self_acc = np.sum(np.array(true_list) == np.array(lstm_self_list)) / length

    acc_list = [bert_acc,  cnn_acc,lstm_self_acc,lstm_acc]
    na_list = ['bert',  'cnn','lstm-self', 'lstm',]

    dic = {}

    for i in range(len(acc_list)):
        for j in range(i + 1, len(acc_list)):
            if acc_list[i] > acc_list[j]:
                dic[na_list[i] + '>' + na_list[j]] = 1
                dic[na_list[j] + '>' + na_list[i]] = 0
            else:
                dic[na_list[i] + '>' + na_list[j]] = 0
                dic[na_list[j] + '>' + na_list[i]] = 1

    return {str(order): dic,'acc_bert':bert_acc,'acc_cnn':cnn_acc,'acc_lstm_self':lstm_self_acc,'lstm':lstm_acc}

'''
count('atis')
count('ag_news')
count('CR')
count('dbpedia')
count('yelp')
count('IMDB')
count('QC')
'''
#count('sst1')
count('mr')
#count('subj')
count('ade')
#count('sms')