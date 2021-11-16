model_name=['cbow','cnn','lstm','lstm-self']
dataset_name=['fdu']


def get_split_dataset_name(dataset_name,model_name):
    path='./'+model_name+'/results/'+dataset_name+'/'
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

import shutil,os
for value in model_name:
    for val in dataset_name:
        split_list=get_split_dataset_name(val,value)
        for v in split_list:
            if value=='cbow':
                if os.path.exists('./'+value+'/results/'+val+'/'+v+'/results/model.pkl'):
                    os.remove('./'+value+'/results/'+val+'/'+v+'/results/model.pkl')
            elif value=='cnn':
                if os.path.exists('./'+value+'/results/'+val+'/'+v+'/results/save/'):
                    shutil.rmtree('./'+value+'/results/'+val+'/'+v+'/results/save/')
            else:
                if os.path.exists('./'+value+'/results/'+val+'/'+v+'/results/results/'):
                    shutil.rmtree('./'+value+'/results/'+val+'/'+v+'/results/results/')




