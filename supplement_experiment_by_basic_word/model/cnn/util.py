import sys

sys.path.append('../../.././fastNLP/')
from fastNLP.core.metrics import MetricBase
import inspect
import warnings
from abc import abstractmethod
from collections import defaultdict
from typing import Union
from copy import deepcopy
import re

import numpy as np
import torch
class AccMetric(MetricBase):


    def __init__(self, pred=None, target=None, seq_len=None):
        r"""

        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_len=seq_len)

        self.total = 0
        self.acc_count = 0

        self.pred_list=[]
        self.target_list=[]

    def evaluate(self, pred, target, seq_len=None):
        r"""
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        # TODO 这里报错需要更改，因为pred是啥用户并不知道。需要告知用户真实的value
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f".")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f".")

        if seq_len is not None and not isinstance(seq_len, torch.Tensor):
            raise TypeError(f".")

        if seq_len is not None and target.dim() > 1:
            max_len = target.size(1)
            masks = seq_len_to_mask(seq_len=seq_len, max_len=max_len)
        else:
            masks = None

        if pred.dim() == target.dim():
            if torch.numel(pred) !=torch.numel(target):
                raise RuntimeError(f"pred have element numbers: {torch.numel(pred)}")

            pass
        elif pred.dim() == target.dim() + 1:
            pred = pred.argmax(dim=-1)
            if seq_len is None and target.dim() > 1:
                warnings.warn("You are not passing `seq_len` to exclude pad when calculate accuracy.")
        else:
            raise RuntimeError(f"In .")

        target = target.to(pred)
        if masks is not None:
            self.acc_count += torch.sum(torch.eq(pred, target).masked_fill(masks.eq(False), 0)).item()
            self.total += torch.sum(masks).item()
        else:
            self.acc_count += torch.sum(torch.eq(pred, target)).item()
            self.total += np.prod(list(pred.size()))

        pred=pred.cpu().numpy().tolist()
        target=target.cpu().numpy().tolist()
        for value in pred:
            self.pred_list.append(value)
        for value in target:
            self.target_list.append(value)

    def get_metric(self, reset=True):
        r"""
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        evaluate_result = {'acc': round(float(self.acc_count) / (self.total + 1e-12), 6)}
        if reset:
            self.acc_count = 0
            self.total = 0

        import pickle
        pickle_file = open('./results/pred.pkl', 'wb')
        pickle.dump(self.pred_list, pickle_file)
        pickle_file.close()

        pickle_file2 = open('./results/target.pkl', 'wb')
        pickle.dump(self.target_list, pickle_file2)
        pickle_file2.close()
        return evaluate_result


def seq_len_to_mask(seq_len, max_len=None):
    r"""

    将一个表示sequence length的一维数组转换为二维的mask，不包含的位置为0。
    转变 1-d seq_len到2-d mask.

    .. code-block::

        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len, max_len=100)
        >>>print(mask.size())
        torch.Size([14, 100])

    :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask




def train(dataset_name,split_dataset_name,text_name='text'):
    # 首先需要加入以下的路径到环境变量，因为当前只对内部测试开放，所以需要手动申明一下路径
    import sys


    import torch.cuda

    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from fastNLP.core.trainer import Trainer
    from fastNLP.core.losses import CrossEntropyLoss
    from fastNLP.core.metrics import AccuracyMetric
    from fastNLP.embeddings import StaticEmbedding

    from fastNLP.core.sampler import BucketSampler
    from fastNLP.core.callback import LRScheduler
    from fastNLP.core.const import Const as C
    from fastNLP.core.vocabulary import VocabularyOption

    from fastNLP.io.pipe.classification import CLSBasePipe
    from fastNLP.io.loader.classification import CLSBaseLoader
    import os
    from fastNLP.io import YelpFullPipe, YelpPolarityPipe

    from fastNLP.core.tester import Tester
    from fastNLP.models.cnn_text_classification import CNNText

    path='../.././dataset/csv_dataset/' + dataset_name + '/' + split_dataset_name + '/'


    class Config():
        datapath = {"train": path+"train.csv", "test": path+"test.csv", "dev": path+"validation.csv"}
        seed = 12345


        train_epoch = 40
        batch_size = 4
        # task = "yelp_f"
        # datadir = 'workdir/datasets/SST'
        # datadir = 'workdir/datasets/yelp_polarity'
        # datadir = 'workdir/datasets/yelp_full'
        # datafile = {"train": "train.txt", "dev": "dev.txt", "test": "test.txt"}
        # datafile = {"train": "train.csv",  "test": "test.csv"}
        lr = 1e-4
        src_vocab_op = VocabularyOption(max_size=100000)

        cls_dropout = 0.1
        weight_decay = 1e-5


    ops = Config()




    # 1.task相关信息：利用dataloader载入dataInfo


    def load_data():
        opt = Config()
        cls_base_loader = CLSBaseLoader()
        cls_base_pipe = CLSBasePipe(lower=True, tokenizer='raw')
        datainfo = cls_base_loader.load(opt.datapath)
        datainfo = cls_base_pipe.process(data_bundle=datainfo)

        # datainfo = YelpFullPipe(lower=True, tokenizer='raw').process_from_file(ops.datapath)
        for ds in datainfo.datasets.values():
            ds.apply_field(len, C.INPUT, C.INPUT_LEN)
            ds.set_input(C.INPUT, C.INPUT_LEN)
            ds.set_target(C.TARGET)

        return datainfo


    datainfo = load_data()
    embedding = StaticEmbedding(datainfo.vocabs['words'], model_dir_or_name='en-word2vec-300d')




    model = CNNText(embed=embedding,num_classes=len(datainfo.vocabs[C.TARGET]))


    loss = CrossEntropyLoss(pred=C.OUTPUT, target=C.TARGET)
    metric = AccuracyMetric(pred=C.OUTPUT, target=C.TARGET)
    optimizer= Adam([param for param in model.parameters() if param.requires_grad==True], lr=ops.lr)

    callbacks = []

    callbacks.append(LRScheduler(CosineAnnealingLR(optimizer, 5)))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    import pathlib
    local_path='./results/'+dataset_name+'/'+split_dataset_name+'/results/'
    pathlib.Path(local_path).mkdir(parents=True,exist_ok=True)




    trainer = Trainer(datainfo.datasets['train'], model, optimizer=optimizer, loss=loss,
                      sampler=BucketSampler(num_buckets=50, batch_size=ops.batch_size),
                      metrics=[metric], use_tqdm=False, save_path=local_path+'save',
                      dev_data=datainfo.datasets['dev'], device=device,
                      check_code_level=-1, batch_size=ops.batch_size, callbacks=callbacks,
                      n_epochs=ops.train_epoch, num_workers=4)


    # distributed trainer
    # trainer = DistTrainer(datainfo.datasets['train'], model, optimizer=optimizer, loss=loss,
    #                       metrics=[metric],
    #                       dev_data=datainfo.datasets['test'], device='cuda',
    #                       batch_size_per_gpu=ops.batch_size, callbacks_all=callbacks,
    #                       n_epochs=ops.train_epoch, num_workers=4)





    print(trainer.train())

    my_metric = AccMetric()


    tester = Tester(datainfo.datasets['test'], model, metrics=my_metric)
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
    if total != None and order != None:
        span = len(split_dataset_name_list) // total
        start = (order - 1) * span
        end = min(start + span, len(split_dataset_name_list))
        if order == total:
            end = len(split_dataset_name_list)
        split_dataset_name_list = split_dataset_name_list[start:end]


    json_path='./results/'+dataset_name+'/'
    from os import listdir
    import os
    if not os.path.exists(json_path):
        json_file_list=[]
    else:
        json_file_list = listdir(json_path)


    json_path='./results/'+dataset_name+'/'
    from os import listdir
    import os
    if not os.path.exists(json_path):
        json_file_list=[]
    else:
        json_file_list = listdir(json_path)





    tem_split_list=[value for value in split_dataset_name_list]


    for value in tem_split_list:

        for val in json_file_list:

            if (value == val) :

                split_dataset_name_list.remove(value)
                break


    if len(split_dataset_name_list)==0:
        print('NONE----------------------')
        return


    for value in split_dataset_name_list:
        print('------' + dataset_name + '-----' + value)
        train(dataset_name, value, text_name)
