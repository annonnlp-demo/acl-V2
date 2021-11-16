from DecisionTree import DTtrain
from knn import knntrain
from lgb import lgbtrain
from RandomForest import RFtrain
from svm import svmtrain

import sys

sys.path.append('../../.././calibration/')

from supplement_experiment.data import get_data
from sklearn.model_selection import train_test_split


def train(target_name, split_list, size,feature_list=[]):
    indicator = 38
    if target_name == 'variance':
        indicator = 38
    elif target_name == 'average':
        indicator = 39
    elif target_name == 'indicator3':
        indicator = 40

    data = get_data(size, split_list)
    X, y = data.iloc[:, 3:33], data.iloc[:, indicator]
    X=X[feature_list]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    rmse1, col1, = DTtrain(X_train, X_test, y_train, y_test, target_name)
    rmse2, col2 = knntrain(X_train, X_test, y_train, y_test, target_name)
    rmse3, col3 = lgbtrain(X_train, X_test, y_train, y_test, target_name)
    rmse4, col4 = svmtrain(X_train, X_test, y_train, y_test, target_name)

    return rmse1, col1, rmse2, col2, rmse3, col3, rmse4, col4,


feature3_list = [


    'grammar_train',
    'grammar_test',



    'ppl_train',
    'ppl_test',
    'test_flesch_reading_ease',
    'test_flesch_reading_ease_propration',

    'train_flesch_reading_ease',
    'train_flesch_reading_ease_propration',


    'd_grammar',

    'd_ppl',
    'd_fre',
    'd_fre_po',
]

feature1_list=['average_length_test',
               'average_length_train',
               'd_length',
               'label_imbalance',
               'label_number',
               ]
feature2_list=['basic_word',
               'basic_word_train',
               'basic_word_test',
               'd_basic_word',
               'language',
               'language_train',
               'language_test',
               'd_language',
               'pmi',
               'pmi_propottion',
               'test_ttr',
               'train_ttr',
               'd_ttr',
               ]

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
list=[feature1_list,feature2_list,feature3_list,feature_name_list_all]
if __name__ == '__main__':
    rmse1_list=[]
    col1_list=[]
    rmse2_list=[]
    col2_list=[]
    rmse3_list=[]
    col3_list=[]
    rmse4_list=[]
    col4_list=[]
    for value in list:
        rmse1, col1, rmse2, col2, rmse3, col3, rmse4, col4, = train('variance',
                                                                    ['right_grammar', 'basic_word', 'right_len'], 987,value)
        rmse1_list.append(rmse1)
        col1_list.append(col1)
        rmse2_list.append(rmse2)
        col2_list.append(col2)
        rmse3_list.append(rmse3)
        col3_list.append(col3)
        rmse4_list.append(rmse4)
        col4_list.append(col4)

    import pandas as pd

    dataframe = pd.DataFrame({'rmse1': rmse1_list,

                              'col1': col1_list,
                              'rmse2': rmse2_list,

                              'col2': col2_list,

                              'rmse3': rmse3_list,

                              'col3': col3_list,
                              'rmse4': rmse4_list,

                              'col4': col4_list,

                              })

    print(dataframe)

    dataframe.to_excel('./var.xls')

    rmse1_list=[]
    col1_list=[]
    rmse2_list=[]
    col2_list=[]
    rmse3_list=[]
    col3_list=[]
    rmse4_list=[]
    col4_list=[]

    for value in list:
        rmse1, col1, rmse2, col2, rmse3, col3, rmse4, col4=train('indicator3', ['right_grammar', 'basic_word', 'right_len'], 987,value)
        rmse1_list.append(rmse1)
        col1_list.append(col1)
        rmse2_list.append(rmse2)
        col2_list.append(col2)
        rmse3_list.append(rmse3)
        col3_list.append(col3)
        rmse4_list.append(rmse4)
        col4_list.append(col4)

    import pandas as pd

    dataframe = pd.DataFrame({'rmse1': rmse1_list,

                              'col1': col1_list,
                              'rmse2': rmse2_list,

                              'col2': col2_list,

                              'rmse3': rmse3_list,

                              'col3': col3_list,
                              'rmse4': rmse4_list,

                              'col4': col4_list,

                              })

    dataframe.to_excel('./svar.xls')
