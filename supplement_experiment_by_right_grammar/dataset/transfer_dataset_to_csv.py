import pandas as pd


def get_split_dataset_name(dataset_name):
    path = './results/'+dataset_name+'/'
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


def convert_dataset_object_to_csv(dataset_name, split_dataset_name, text_name='text'):
    path = './results/'+dataset_name+'/'+split_dataset_name+'/'
    import os
    if not os.path.exists(path):
        os.makedirs(path)

    from datasets import load_from_disk
    train_dataset = load_from_disk(path + 'train')
    test_dataset = load_from_disk(path + 'test')
    dev_dataset = load_from_disk(path + 'validation')

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
        word_list = test_dataset.features['label'].names

    train_data_list = train_dataset[text_name]
    train_data_list=[value.rstrip() for value in train_data_list]
    train_label_list = train_dataset['label']
    train_label_list = [word_list[int(i)] for i in train_label_list]

    test_data_list = test_dataset[text_name]
    test_data_list=[value.rstrip() for value in test_data_list]
    test_label_list = test_dataset['label']
    test_label_list = [word_list[int(i)] for i in test_label_list]

    dev_data_list = dev_dataset[text_name]
    dev_data_list=[value.rstrip() for value in dev_data_list]
    dev_label_list = dev_dataset['label']
    dev_label_list = [word_list[int(i)] for i in dev_label_list]

    dataset_save_path =  './csv_dataset/' + dataset_name + '/' + split_dataset_name + '/'
    import pathlib
    pathlib.Path(dataset_save_path).mkdir(parents=True,exist_ok=True)






    dataframe = pd.DataFrame({'l': train_label_list, 's': train_data_list, })
    dataframe.to_csv(dataset_save_path + 'train.csv', index=False, sep=',', header=False, quoting=1)

    dataframe = pd.DataFrame({'l': test_label_list, 's': test_data_list, })
    dataframe.to_csv(dataset_save_path + 'test.csv', index=False, sep=',', header=False, quoting=1)

    dataframe = pd.DataFrame({'l': dev_label_list, 's': dev_data_list, })
    dataframe.to_csv(dataset_save_path + 'validation.csv', index=False, sep=',', header=False, quoting=1)


def convert(dataset_name, text_name='text'):
    list = get_split_dataset_name(dataset_name)

    for value in list:
        convert_dataset_object_to_csv(dataset_name, split_dataset_name=value, text_name=text_name)

convert('ag', )

convert('ade')
convert('imdb', )
convert('MR', )

convert('rotten', )



convert('fdu', 'content')
convert('sst1', )
convert('sst2', )
convert('subj', )




convert('dbpedia', 'content')

convert('yelp', )
