from pathlib import Path
import numpy as np
import sys

sys.path.append('../.././calibration/')

from supplement_experiment.util import *

feature_name_list_all = [
    'average_length_test',
    'average_length_train',
    'basic_word',
    'basic_word_train',
    'basic_word_test',
    'grammar_train',
    'grammar_test',
    'label_imbalance',
    'label_number',
    'language',
    'language_train',
    'language_test',
    'pmi',
    'pmi_propottion',
    'ppl_train',
    'ppl_test',
    'test_flesch_reading_ease',
    'test_flesch_reading_ease_propration',
    'test_ttr',
    'train_flesch_reading_ease',
    'train_flesch_reading_ease_propration',
    'train_ttr',
    'd_ttr',
    'd_length',
    'd_basic_word',
    'd_grammar',
    'd_language',
    'd_ppl',
    'd_fre',
    'd_fre_po',
]
#d_fre
#






abb_feature_name_list = [
    'av_len_te',
    'av_len_tr',
    'ba_wo',
    'ba_wo_tr',
    'ba_wo_te',
    'gra_tra',
    'gra_tes',
    'lab_imba',
    'lab_num',
    'lang',
    'lang_tr',
    'lang_te',
    'pmi',
    'pmi_pro',
    'ppl_tr',
    'ppl_te',
    'te_f_r_e',
    'te_f_r_e_p',
    'te_ttr',
    'tr_f_r_e',
    'tr_f_r_e_pn',
    'tr_ttr',
    'd_ttr',
    'd_len',
    'd_ba_wo',
    'd_gram',
    'd_lang',
    'd_ppl',
    'd_fre',
    'd_fre_po',
]

columns=['dataset', 'split_name', 'split_type'] + feature_name_list_all + ['bert','cbow','cnn','lstm','lstm_self','stdev','average','indicator5']

feature_name_dic = {
}
print(len(feature_name_list_all))
print(len(abb_feature_name_list))
for index in range(len(feature_name_list_all)):
    feature_name_dic[feature_name_list_all[index]] = abb_feature_name_list[index]

#print(feature_name_dic)
#target_name = 'variance'

import math

def get_data(size,split_type_list=None):

    if split_type_list is None:
        split_type_list = ['basic_word']
    results_list=[]

    for value in split_type_list:
        results_list.append(get_data_(value))




    results = {}
    import pandas as pd

    for value in columns:
        results[value]=[]
        for res in results_list:

            results[value] += res[value]

    data = pd.DataFrame.from_dict(results)

    #data.rename(columns=feature_name_dic, inplace=True)


    if size>987:
        size=987

    print(size)
    print(len(data))
    results=data.sample(n=size,replace=False,random_state=1)

    return results


def get_data_(json_file_type='gram'):
    import pandas as pd

    json_file_path = '/colabration/supplement_experiment/results/' + json_file_type + "-results.json"

    dic = get_json(json_file_path)

    results = {}

    for value in columns:
        results[value] = []

    results['stdev']=[]
    results['average']=[]
    results['indicator5']=[]

    for k, v in dic.items():


        for key, value in v.items():
            results['dataset'].append(k)
            results['split_name'].append(key)

            results['split_type'].append(json_file_type)

            for feature in feature_name_list_all:
                if feature=='d_ttr':
                    results[feature].append(math.pow((value['train_ttr']-value['test_ttr'])/(value['train_ttr']),2))
                elif feature=='d_fre':
                    results[feature].append(math.pow((value['train_flesch_reading_ease']-value['test_flesch_reading_ease'])/(value['train_flesch_reading_ease']),2))
                elif feature=='d_length':

                    results[feature].append(math.pow((value['average_length_train']-value['average_length_test'])/(value['average_length_train']),2))


                elif feature=='d_basic_word':
                    if value['basic_word_train']==0.0:
                        results[feature].append(0)
                    else:
                        results[feature].append(math.pow((value['basic_word_train']-value['basic_word_test'])/(value['basic_word_train']),2))
                elif feature=='d_grammar':
                    if value['grammar_train']==0:
                        results[feature].append(0)
                    else:
                        results[feature].append(math.pow((value['grammar_train']-value['grammar_test'])/(value['grammar_train']),2))
                elif feature=='d_language':
                    if value['language_train']==0.0:
                        results[feature].append(0)
                    else:
                        results[feature].append(math.pow((value['language_train']-value['language_test'])/(value['language_train']),2))
                elif feature=='d_ppl':
                    results[feature].append(math.pow((value['ppl_train']-value['ppl_test'])/(value['ppl_train']),2))
                elif feature=='d_fre_po':
                    results[feature].append(math.pow((value['train_flesch_reading_ease_propration']-value['test_flesch_reading_ease_propration'])/(value['train_flesch_reading_ease_propration']),2))
                else:

                    results[feature].append(value[feature])

            results['bert'].append(value['bert_acc'])
            results['cbow'].append(value['cbow_acc'])
            results['cnn'].append(value['cnn_acc'])
            results['lstm'].append(value['lstm_acc'])
            results['lstm_self'].append(value['lstm_self_acc'])

            import statistics

            #variance=statistics.pvariance([value['bert_acc']*100,value['cbow_acc']*100,value['cnn_acc']*100,value['lstm_acc']*100,value['lstm_self_acc']*100])
            variance=np.std([value['bert_acc']*100,value['cnn_acc']*100,value['lstm_acc']*100,value['lstm_self_acc']*100],ddof=1)
            results['stdev'].append(variance)
            results['average'].append(statistics.mean([value['bert_acc'],value['cnn_acc'],value['lstm_acc'],value['lstm_self_acc']])*100)
            mean=statistics.mean([value['bert_acc']*100,value['cnn_acc']*100,value['lstm_acc']*100,value['lstm_self_acc']*100])
            results['indicator5'].append(variance*(100-mean))


    return results

'''
dic = {
    'average_length': 'a_len',
    'basic_word': 'ba_word',
    'grammar': 'grammar',
    'label_imbalance': 'lab_imba',
    'label_number': 'lab_num',
    'language': 'langu',
    'pmi': 'pmi',
    'pmi_propottion': 'pmi_pro',
    'ppl': 'ppl',
    'test_flesch_reading_ease': 'te_f_r_e',
    'test_flesch_reading_ease_propration': 'te_f_r_e_p',
    'test_ttr': 'te_ttr',
    'train_flesch_reading_ease': 'tr_f_r_e',
    'train_flesch_reading_ease_propration': 'tr_f_r_e_p',
    'train_ttr': 'tr_ttr',
    'd_ttr':'d_ttr',
    'd_fre':'d_fre',
    }
'''


#df=df[['indicator5']]
