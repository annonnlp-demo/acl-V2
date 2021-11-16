


import sys

sys.path.append('../../.././fastNLP/')
from fastNLP.core.metrics import MetricBase
import os


class AccMetric(MetricBase):

    def __init__(self):
        super().__init__()
        # 根据你的情况自定义指标
        self.total = 0
        self.acc_count = 0
        self.pred_list = []
        self.target_list = []

    # evaluate的参数需要和DataSet 中 field 名以及模型输出的结果 field 名一致，不然找不到对应的value
    # pred, target 的参数是 fastNLP 的默认配置
    def evaluate(self, pred, target):
        # dev或test时，每个batch结束会调用一次该方法，需要实现如何根据每个batch累加metric
        self.total += target.size(0)
        self.acc_count += target.eq(pred).sum().item()

        pred = pred.cpu().numpy().tolist()
        target = target.cpu().numpy().tolist()
        for value in pred:
            self.pred_list.append(value)
        for value in target:
            self.target_list.append(value)

    def get_metric(self, reset=True):  # 在这里定义如何计算metric
        acc = self.acc_count / self.total
        if reset:  # 是否清零以便重新计算
            self.acc_count = 0
            self.total = 0

        import pickle
        pickle_file = open('./results/pred.pkl', 'wb')
        pickle.dump(self.pred_list, pickle_file)
        pickle_file.close()

        pickle_file2 = open('./results/target.pkl', 'wb')
        pickle.dump(self.target_list, pickle_file2)
        pickle_file2.close()

        return {'acc': acc}
        # 需要返回一个dict，key为该metric的名称，该名称会显示到Trainer的progress bar中

import torch.nn as nn
from fastNLP.core.const import Const as C
from fastNLP.modules.encoder.lstm import LSTM
from fastNLP.embeddings.utils import get_embeddings
from fastNLP.modules.attention import SelfAttention
from fastNLP.modules.decoder.mlp import MLP


class BiLSTM_SELF_ATTENTION(nn.Module):
    def __init__(self, init_embed,
                 num_classes,
                 hidden_dim=256,
                 num_layers=1,
                 attention_unit=256,
                 attention_hops=1,
                 nfc=128,save_or_not=False):
        super(BiLSTM_SELF_ATTENTION,self).__init__()
        self.embed = get_embeddings(init_embed)
        self.lstm = LSTM(input_size=self.embed.embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True)
        self.attention = SelfAttention(input_size=hidden_dim * 2 , attention_unit=attention_unit, attention_hops=attention_hops)
        self.mlp = MLP(size_layer=[hidden_dim* 2*attention_hops, nfc, num_classes])
        self.save_or_not=save_or_not
        self.results=[]

    def forward(self, words):
        x_emb = self.embed(words)
        output, _ = self.lstm(x_emb)
        after_attention, penalty = self.attention(output,words)
        after_attention =after_attention.view(after_attention.size(0),-1)
        output = self.mlp(after_attention)
        if self.save_or_not:
            result=to_list(output)
            for value in result:
                self.results.append(value)
        return {C.OUTPUT: output}

    def predict(self, words):
        output = self(words)
        _, predict = output[C.OUTPUT].max(dim=1)
        return {C.OUTPUT: predict}

    def save(self,save):
        print('change to get pro:')
        print(save)
        self.save_or_not=save

    def get(self):
        return self.results


def softmax(num_list):
    import numpy as np
    e_list = np.exp(num_list)
    max_num = max(e_list)
    return max_num / (sum(e_list))


def to_list(tensor):
    list = tensor.cpu().numpy().tolist()
    result = []
    for value in list:
        result.append(softmax(value))
    return result


def train(dataset_name, split_dataset_name,number_class, text_name='text'):
    from fastNLP.io.pipe.classification import IMDBPipe
    from fastNLP.embeddings import StaticEmbedding

    from fastNLP.core.losses import CrossEntropyLoss
    from fastNLP.core.metrics import AccuracyMetric
    from fastNLP.core.trainer import Trainer
    from torch.optim import Adam
    from fastNLP.io.pipe.classification import CLSBasePipe
    from fastNLP.io.loader.classification import CLSBaseLoader
    from fastNLP.core.tester import Tester
    import pickle


    import pathlib
    local_path='./results/'+dataset_name+'/'+split_dataset_name+'/results/'
    pathlib.Path(local_path).mkdir(parents=True,exist_ok=True)


    path='../.././dataset/csv_dataset/' + dataset_name + '/' + split_dataset_name + '/'

    class Config():
        train_epoch = 30
        lr = 0.001

        num_classes = number_class
        hidden_dim = 256
        num_layers = 1
        attention_unit = 256
        attention_hops = 1
        nfc = 128

        task_name = "CR"
        datapath = {"train": path+"train.csv", "test": path+"test.csv", "dev": path+"validation.csv"}
        save_model_path = local_path+"/results/"
        device_number = 0
        model_save_name = local_path+'/model.pkl'


    def train__(data_bundle, model, optimizer, loss, metrics, opt):
        trainer = Trainer(data_bundle.datasets['train'], model, optimizer=optimizer, loss=loss,
                          metrics=metrics, dev_data=data_bundle.datasets['dev'], device=opt.device_number,
                          check_code_level=-1,
                          n_epochs=opt.train_epoch, save_path=opt.save_model_path,batch_size=16)
        trainer.train()

    def save_model(model, model_save_name):
        pickle_file = open(model_save_name, 'wb')
        print('save model')
        pickle.dump(model, pickle_file)
        pickle_file.close()

    opt = Config()

    # load data
    # data_bundle=IMDBPipe.process_from_file(opt.datapath)
    cls_base_loader = CLSBaseLoader()
    cls_base_pipe = CLSBasePipe()
    data_bundle = cls_base_loader.load(opt.datapath)
    data_bundle = cls_base_pipe.process(data_bundle=data_bundle)
    # print(data_bundle.datasets["train"])
    # print(data_bundle)

    # define model
    vocab = data_bundle.vocabs['words']
    embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-840b-300', requires_grad=True)
    model=BiLSTM_SELF_ATTENTION(init_embed=embed, num_classes=opt.num_classes, hidden_dim=opt.hidden_dim, num_layers=opt.num_layers, attention_unit=opt.attention_unit, attention_hops=opt.attention_hops, nfc=opt.nfc)


# define loss_function and metrics
    loss = CrossEntropyLoss()
    metrics = AccuracyMetric()
    optimizer = Adam([param for param in model.parameters() if param.requires_grad == True], lr=opt.lr)

    train__(data_bundle, model, optimizer, loss, metrics, opt)

    my_metric = AccMetric()

    model.save(True)
    tester = Tester(data_bundle.get_dataset('test'), model, metrics=my_metric)
    results=tester.test()




    import json

    dic = {'dataset_name': dataset_name, 'split_name': split_dataset_name, 'results': results}
    json_str = json.dumps(dic, indent=1)

    with open(local_path + 'results.json', 'w') as json_file:
        json_file.write(json_str)



def get_split_dataset_name(dataset_name):
    path = '../.././dataset/csv_dataset/' + dataset_name + '/'
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


def train_(dataset_name, num_labels_size, text_name='text', total=None, order=None):
    split_dataset_name_list = get_split_dataset_name(dataset_name)

    tem_split_list=[value for value in split_dataset_name_list]

    import os
    for value in tem_split_list:
        if os.path.exists('./results/fdu/'+value+'/results/results.json'):
            split_dataset_name_list.remove(value)
            print(value)

    if total != None and order!=None:
        total_length=len(split_dataset_name_list)
        span= total_length // total
        start= (order - 1) * span
        end=min(start+span,total_length)
        if order==total:
            end=total_length
        split_dataset_name_list=split_dataset_name_list[start:end]



    for value in split_dataset_name_list:
        print('------' + dataset_name + '-----' + value)
        train(dataset_name, value, num_labels_size,text_name)


def train__(dataset_name, num_labels_size, split_name, text_name='text', total=None, order=None):

    print('------' + dataset_name + '-----' + split_name)
    train(dataset_name, split_name, num_labels_size,text_name)

