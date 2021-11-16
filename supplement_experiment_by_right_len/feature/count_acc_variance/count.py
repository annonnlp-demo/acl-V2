
from pathlib import Path

from supplement_experiment_by_gram.feature.util.util import *
bert_json='.././count_acc_bert/results/bert-acc.json'
cbow_json='.././count_acc_cbow/results/cbow-acc.json'
cnn_json='.././count_acc_cnn/results/cnn-acc.json'
lstm_json='.././count_acc_lstm/results/lstm-acc.json'
lstm_self_json='.././count_acc_lstm_self/results/lstm-self-acc.json'


def count_accuracy():

    bert_dic=get_json(bert_json)

    cbow_dic=get_json(cbow_json)
    cnn_dic=get_json(cnn_json)
    lstm_dic=get_json(lstm_json)
    lstm_self_dic=get_json(lstm_self_json)

    results={}

    for k,v in bert_dic.items():
        tem={}
        for ke,va in v.items():
            if bert_dic[k][ke]==None or cbow_dic[k][ke]==None or cnn_dic[k][ke]==None or lstm_dic[k][ke] ==None or lstm_self_dic[k][ke]==None:
                print(k)
                print(ke)

                continue
            tem_bert=bert_dic[k][ke]
            tem_cbow=cbow_dic[k][ke]
            tem_cnn=cnn_dic[k][ke]
            tem_lstm=lstm_dic[k][ke]
            tem_lstm_self=lstm_self_dic[k][ke]



            import numpy as np
            arr = [tem_bert,tem_cnn,tem_cbow,tem_lstm,tem_lstm_self]

            arr_std = np.std(arr, ddof=1)
            tem[ke]=arr_std

        results[k]=tem

    json_root='./results/'
    Path(json_root).mkdir(parents=True,exist_ok=True)
    json_path=json_root+'acc-variance.json'
    save_json(results,json_path)


count_accuracy()














