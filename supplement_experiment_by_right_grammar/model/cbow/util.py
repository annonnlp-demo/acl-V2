import torch
from torch import nn


def evaluate_validation(scores, loss_function, gold):
    guesses = scores.argmax(dim=1)
    n_correct = (guesses == gold).sum().item()
    guesses_list = guesses.cpu().cpu().numpy().tolist()
    gold_list = gold.cpu().cpu().numpy().tolist()
    return n_correct, loss_function(scores, gold).item(), guesses_list, gold_list


class CBoWTextClassifier2(torch.nn.Module):

    def __init__(self, text_field, class_field, emb_dim, n_hidden=10, dropout=0.5):
        super().__init__()
        voc_size = len(text_field.vocab)
        n_classes = len(class_field.vocab)
        self.embedding = nn.Embedding(voc_size, emb_dim)
        self.hidden_layer = nn.Linear(emb_dim, n_hidden)
        self.top_layer = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, texts):
        embedded = self.embedding(texts)
        cbow = embedded.mean(dim=0)
        cbow_drop = self.dropout(cbow)
        hidden = torch.relu(self.hidden_layer(cbow_drop))
        scores = self.top_layer(hidden)
        return scores


def train(num_labels_size, dataset_name, split_dataset_name, text_name='text'):
    import pathlib
    path = './results/' + dataset_name + '/' + split_dataset_name + '/results/'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    dataset_path = '../.././dataset/results/' + dataset_name + '/' + split_dataset_name + '/'

    import torch

    import torchtext
    from collections import defaultdict
    import time
    import os
    import sys

    sys.path.append('../../../')

    class Config():
        train_epoch = 200
        batch_size = 16
        lr = 0.001
        patience = 20

    def save_model(model, model_save_name):
        import pickle
        pickle_file = open(model_save_name, 'wb')
        print('save model')
        pickle.dump(model, pickle_file)
        pickle_file.close()

    def get_object(filename):
        pickle_file = open(filename, 'rb')
        array = pickle.load(pickle_file)
        pickle_file.close()
        return array

    # change
    def get_train_dataset(datafields, path, dataset_name):
        opt = Config()

        import torchtext
        from datasets import load_from_disk
        _dataset = load_from_disk(path + 'train/')

        if dataset_name == 'dbpedia':
            word_list = [
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
            word_list = _dataset.features['label'].names

        examples = []
        for value in _dataset:
            doc = value[text_name]
            label = word_list[int(value['label'])]
            examples.append(torchtext.data.Example.fromlist([doc, label], datafields))
        return torchtext.data.Dataset(examples, datafields)

    def get_test_dataset(datafields, path, dataset_name):
        opt = Config()

        import torchtext
        from datasets import load_from_disk
        _dataset = load_from_disk(path + 'test/')

        if dataset_name == 'dbpedia':
            word_list = [
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
            word_list = _dataset.features['label'].names

        examples = []
        for value in _dataset:
            doc = value[text_name]
            label = word_list[int(value['label'])]
            examples.append(torchtext.data.Example.fromlist([doc, label], datafields))
        return torchtext.data.Dataset(examples, datafields)

    def get_vali_dataset(datafields, path, dataset_name):
        opt = Config()

        import torchtext
        from datasets import load_from_disk
        _dataset = load_from_disk(path + 'validation/')

        if dataset_name == 'dbpedia':
            word_list = [
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
            word_list = _dataset.features['label'].names

        examples = []
        for value in _dataset:
            doc = value[text_name]
            label = word_list[int(value['label'])]
            examples.append(torchtext.data.Example.fromlist([doc, label], datafields))
        return torchtext.data.Dataset(examples, datafields)

    opt = Config()
    # We first declare the fields of the dataset: one field for the text, and one for the output label.
    # For the text field, we also provide a tokenizer.
    # In this case, we can use a simple tokenizer since the text is already tokenized in the file.
    TEXT = torchtext.data.Field(sequential=True, tokenize=lambda x: x.split())
    LABEL = torchtext.data.LabelField(is_target=True)
    datafields = [('text', TEXT), ('label', LABEL)]

    train = get_train_dataset(datafields, dataset_path, dataset_name)
    test = get_test_dataset(datafields, dataset_path, dataset_name)
    valid = get_vali_dataset(datafields, dataset_path, dataset_name)
    # Build vocabularies from the dataset.
    TEXT.build_vocab(train, valid, test, max_size=10000)
    LABEL.build_vocab(train, valid, test)

    import pickle

    pickle_file_voca = open(path + 'voca_list.pkl', 'wb')
    pickle.dump(LABEL.vocab.itos, pickle_file_voca)
    pickle_file_voca.close()

    # Declare the model. We'll use the shallow CBoW classifier or the one that has one hidden layer.
    model = CBoWTextClassifier2(TEXT, LABEL, emb_dim=16)
    # model = CBoWTextClassifier2(TEXT, LABEL, emb_dim=16)

    # Put the model on the device.
    device = 'cuda'
    model.to(device)

    # The BucketIterator groups sentences of similar lengths into "buckets", which reduces the need
    # for padding when we create minibatches.
    # See here: https://pytorch.org/text/data.html#torchtext.data.BucketIterator
    train_iterator = torchtext.data.BucketIterator(
        train,
        device=device,
        batch_size=16,
        sort_key=lambda x: len(x.text),
        repeat=False,
        train=True)

    valid_iterator = torchtext.data.Iterator(
        valid,
        device=device,
        batch_size=16,
        repeat=False,
        train=False,
        sort=False)

    test_iterator = torchtext.data.Iterator(
        test,
        device=device,
        batch_size=32,
        repeat=False,
        train=False,
        sort=False)

    # Cross-entropy loss as usual, since we have a classification problem.
    loss_function = torch.nn.CrossEntropyLoss()

    # Adam optimizer. We can try to tune the learning rate to get a fast convergence while avoiding instability.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # To speed up training, we'll put all the batches onto the GPU. This will avoid repeating
    # the preprocessing of the batches as well as the GPU communication overhead.
    # We can do this because the dataset is not that big so that it fits in the GPU memory.
    # Torchtext will handle all administration: mapping text to integers and putting everything into tensors.
    train_batches = list(train_iterator)
    valid_batches = list(valid_iterator)
    test_batches = list(test_iterator)

    # We'll keep track of some indicators and plot them in the end.
    history = defaultdict(list)

    best_accuracy = 0
    pred_list = []
    true_list = []
    best_epoch = 0

    print(opt.train_epoch)
    print(opt.batch_size)

    for i in range(200):

        pred_temp_list = []
        true_temp_list = []

        t0 = time.time()

        loss_sum = 0
        n_batches = 0

        # Calling model.train() will enable the dropout layers.
        model.train()

        # We iterate through the batches created by torchtext.
        # For each batch, we can access the text part and the output label part separately.
        for batch in train_batches:
            # Compute the output scores.
            scores = model(batch.text)
            # Then the loss function.
            loss = loss_function(scores, batch.label)

            # Compute the gradient with respect to the loss, and update the parameters of the model.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            n_batches += 1

        train_loss = loss_sum / n_batches
        history['train_loss'].append(train_loss)

        # After each training epoch, we'll compute the loss and accuracy on the validation set.
        n_correct = 0
        n_valid = len(valid)
        loss_sum = 0
        n_batches = 0

        # Calling model.train() will disable the dropout layers.
        model.eval()

        for batch in valid_batches:
            scores = model(batch.text)
            n_corr_batch, loss_batch, guess_list, true_1_list = evaluate_validation(scores, loss_function, batch.label)
            loss_sum += loss_batch
            n_correct += n_corr_batch
            n_batches += 1
        val_acc = n_correct / n_valid
        val_loss = loss_sum / n_batches

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = i
            save_model(model, path + 'model.pkl')

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        t1 = time.time()

        print(
            f'Epoch {i + 1}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, '
            f'val acc: {val_acc:.4f}, time = {t1 - t0:.4f}')

        if i - best_epoch >= opt.patience:
            break

    print(
        f'best:Epoch {best_epoch + 1}: val acc: {best_accuracy:.4f}')
    import pickle

    model = get_object(path + 'model.pkl')

    model.eval()
    n_correct = 0
    n_valid = len(test)
    loss_sum = 0
    n_batches = 0
    pred_temp_list = []
    true_temp_list = []

    for batch in test_batches:
        scores = model(batch.text)
        n_corr_batch, loss_batch, guess_list, true_1_list = evaluate_validation(scores, loss_function, batch.label)
        loss_sum += loss_batch
        n_correct += n_corr_batch
        n_batches += 1
        for value in range(len(guess_list)):
            pred_temp_list.append(guess_list[value])
            true_temp_list.append(true_1_list[value])
    test_acc = n_correct / n_valid

    import json

    dic = {'dataset_name': dataset_name, 'split_name': split_dataset_name, 'test accuracy': test_acc,
           'bestEpoch': best_epoch + 1, "val acc": best_accuracy
           }
    json_str = json.dumps(dic, indent=1)

    with open(path + 'results.json', 'w') as json_file:
        json_file.write(json_str)


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


def train_(dataset_name, num_labels_size, text_name='text', total=None, order=None):
    split_dataset_name_list = get_split_dataset_name(dataset_name)

    if total != None and order != None:
        span = len(split_dataset_name_list) // total
        start = (order - 1) * span
        end = min(start + span, len(split_dataset_name_list))
        if order == total:
            end = len(split_dataset_name_list)
        split_dataset_name_list = split_dataset_name_list[start:end]

    for value in split_dataset_name_list:
        print('------' + dataset_name + '-----' + value)
        train(num_labels_size, dataset_name, value, text_name)
