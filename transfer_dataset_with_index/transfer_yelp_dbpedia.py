from datasets import load_dataset
from datasets import load_from_disk
from datasets import concatenate_datasets
import os, string, json
import datasets
from datasets.features import Value, ClassLabel

feature_list = []


def get_length(text):
    return text.count(' ') + 1


def get_feature(dataset_name):
    from datasets import Features, Dataset
    import datasets
    if dataset_name == 'yelp':
        names = ['1', '2']
        features = Features(
            {

                "label": datasets.features.ClassLabel(names=names, num_classes=len(names)),
                "text": datasets.Value("string"),
                "index_raw": datasets.Value("int64")

            }
        )
    elif dataset_name == 'dbpedia':
        names = [
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
        features = Features(
            {
                "label": datasets.features.ClassLabel(names=names, num_classes=len(names)),
                "content": datasets.Value("string"),
                "index_raw": datasets.Value("int64")

            }
        )
    elif dataset_name == 'ag':
        features = Features(
            {'label': ClassLabel(num_classes=4, names=['World', 'Sports', 'Business', 'Sci/Tech'], names_file=None,
                                 id=None), 'text': Value(dtype='string', id=None),
             "index_raw": datasets.Value("int64")}
        )
    elif dataset_name == 'imdb':
        features = Features(
            {'label': ClassLabel(num_classes=2, names=['neg', 'pos'], names_file=None, id=None),
             'text': Value(dtype='string', id=None),
             "index_raw": datasets.Value("int64")}
        )
    elif dataset_name == 'fdu':
        features = Features(
            {'content': Value(dtype='string', id=None),
             'label': ClassLabel(num_classes=2, names=['0', '1'], names_file=None, id=None),
             "index_raw": datasets.Value("int16")}
        )
    elif dataset_name == 'sst2':
        features = Features(
            {'label': ClassLabel(num_classes=2, names=['0', '1'], names_file=None, id=None),
             'text': Value(dtype='string', id=None),
             "index_raw": datasets.Value("int64")}
        )
    elif dataset_name == 'ade':
        features = Features(
            {'label': ClassLabel(num_classes=2, names=['Not-Related', 'Related'], names_file=None, id=None),
             'text': Value(dtype='string', id=None),
             "index_raw": datasets.Value("int16")}
        )
    elif dataset_name == 'rotten':
        features = Features(
            {'label': ClassLabel(num_classes=2, names=['neg', 'pos'], names_file=None, id=None),
             'text': Value(dtype='string', id=None),
             "index_raw": datasets.Value("int16")}
        )

    elif dataset_name == 'subj':
        features = Features(
            {'label': ClassLabel(num_classes=2, names=['0', '1'], names_file=None, id=None),
             'text': Value(dtype='string', id=None),
             "index_raw": datasets.Value("int16")}
        )
    elif dataset_name == 'sst1':
        features = Features(
            {'label': ClassLabel(num_classes=5, names=['0', '1', '2', '3', '4'], names_file=None, id=None),
             'text': Value(dtype='string', id=None),
             "index_raw": datasets.Value("int16")}
        )

    elif dataset_name == "MR":
        features = Features(
            {'label': ClassLabel(num_classes=2, names=['0', '1'], names_file=None, id=None),
             'text': Value(dtype='string', id=None),
             "index_raw": datasets.Value("int16")}
        )

    return features


def convert_(dataset_name, text_name='text'):
    train_dataset = load_from_disk('.././dataset/' + dataset_name + '/results/train/')
    test_dataset = load_from_disk('.././dataset/' + dataset_name + '/results/test/')
    dev_dataset = load_from_disk('.././dataset/' + dataset_name + '/results/validation/')

    train_dataset = train_dataset.flatten_indices()
    test_dataset = test_dataset.flatten_indices()
    dev_dataset = dev_dataset.flatten_indices()

    dataset = concatenate_datasets([train_dataset, test_dataset])
    dataset = concatenate_datasets([dataset, dev_dataset])

    text_list = []
    label_list = []
    index_list = []

    for index in range(len(dataset)):
        item = dataset[index]
        text_list.append(item[text_name])
        label_list.append(item['label'])
        index_list.append(index)

    dic = {}
    dic[text_name] = text_list
    dic['label'] = label_list
    dic['index_raw']=index_list

    from datasets import Features, Dataset
    features = get_feature(dataset_name)
    new_dataset = Dataset.from_dict(dic, features=features)

    from pathlib import Path
    path = './results/' + dataset_name + '/'
    Path(path).mkdir(parents=True, exist_ok=True)

    new_dataset.save_to_disk(path)


