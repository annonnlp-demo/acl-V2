def train(num_labels_size ,dataset_name,split_dataset_name,text_name='text'):
    from transformers import BertForSequenceClassification, BertTokenizerFast, \
        Trainer, TrainingArguments

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import pickle

    import gc

    # change
    num_labels_size = num_labels_size
    # change
    batch_size = 8
    # change
    model_checkpoint = 'bert-base-uncased'
    # change
    number_train_epoch = 5
    # change
    column_label_name = 'label'

    # torch.backends.cudnn.enabled = False




    # change
    def tokenize(batch):
        return tokenizer(batch[text_name], truncation=True, padding=True)


    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        # change
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


    def change_transformers_dataset_2_right_format(dataset, label_name):
        return dataset.map(lambda example: {'label': example[label_name]}, remove_columns=[label_name])

    path='../.././dataset/results/'+dataset_name+'/'+split_dataset_name+'/'
    # change
    def get_train_dataset():
        from datasets import load_from_disk
        dataset = load_from_disk(path+'train')
        return dataset
    # change
    def get_test_dataset():
        from datasets import load_from_disk
        dataset = load_from_disk(path+'test')
        return dataset

    def get_dev_dataset():
        from datasets import load_from_disk
        dataset = load_from_disk(path+'validation')
        return dataset


    model = BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels_size)
    tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint, use_fast=True)

    # change
    test_dataset = get_test_dataset()
    test_dataset=test_dataset.shuffle()
    train_dataset = get_train_dataset()
    train_dataset=train_dataset.shuffle()
    dev_dataset = get_dev_dataset()
    dev_dataset=dev_dataset.shuffle()


    train_dataset = change_transformers_dataset_2_right_format(train_dataset, column_label_name)
    test_dataset = change_transformers_dataset_2_right_format(test_dataset, column_label_name)
    dev_dataset = change_transformers_dataset_2_right_format(dev_dataset, column_label_name)


    train_encoded_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    test_encoded_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))
    dev_encoded_dataset = dev_dataset.map(tokenize, batched=True, batch_size=len(dev_dataset))

    del train_dataset
    del test_dataset
    gc.collect()

    # change
    train_encoded_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label', 'token_type_ids'])
    test_encoded_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label', 'token_type_ids'])

    dev_encoded_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label', 'token_type_ids'])

    import pathlib
    path='./results/'+dataset_name+'/'+split_dataset_name+'/results/'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=path,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=number_train_epoch,
        weight_decay=0.01,
        do_predict=True,
        warmup_steps=500,
        logging_dir='./logs',
        save_total_limit=1,
        load_best_model_at_end=True,
        # metric_for_best_model=compute_metrics,
    )

    trainer = Trainer(
        model=model,
        args=args,
        compute_metrics=compute_metrics,
        train_dataset=train_encoded_dataset,
        eval_dataset=dev_encoded_dataset,
        tokenizer=tokenizer
    )
    print('dataset load close')
    print('start train')

    trainer.train()
    print('start test')
    trainer.evaluate(test_encoded_dataset)

    import pathlib
    pathlib.Path(path+'model').mkdir(parents=True, exist_ok=True)
    trainer.save_model(path+'model')
    prediction, label_ids, metrics = trainer.predict(test_dataset=test_encoded_dataset)





    import json

    dic={'dataset_name':dataset_name,'split_name':split_dataset_name,'results':metrics}
    json_str = json.dumps(dic,indent=1)

    with open(path+'results.json', 'w') as json_file:
        json_file.write(json_str)


def get_split_dataset_name(dataset_name):
    path='../.././dataset/results/'+dataset_name+'/'
    from os import listdir
    from os.path import join
    list=listdir(path)
    if 'test' in list:
        list.remove('test')
    if 'train' in list:
        list.remove('train')
    if 'validation' in list:
        list.remove('validation')
    if '__init__.py' in list:
        list.remove('__init__.py')

    return list


def train_(dataset_name,num_labels_size, text_name='text',total=None,order=None):

    split_dataset_name_list=get_split_dataset_name(dataset_name)

    if total!=None and order!=None:
        span=len(split_dataset_name_list)//total
        start=(order-1)*span
        end=min(start+span,len(split_dataset_name_list))
        if order==total:
            end=len(split_dataset_name_list)
        split_dataset_name_list=split_dataset_name_list[start:end]


    for value in split_dataset_name_list:
        print('------'+dataset_name+'-----'+value)
        train(num_labels_size,dataset_name,value,text_name)
