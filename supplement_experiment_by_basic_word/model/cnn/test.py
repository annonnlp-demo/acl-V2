
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


    print(split_dataset_name_list)
    print(json_file_list)


    tem_split_list=[value for value in split_dataset_name_list]
    print(tem_split_list)

    for value in tem_split_list:
        print(value)
        for val in json_file_list:
            print(val)
            if (value == val) :
                print('----------TRUE---------')
                split_dataset_name_list.remove(value)
                break


    print(dataset_name)
    print(split_dataset_name_list)
    print(len(split_dataset_name_list))




train_('sst1',5,)
