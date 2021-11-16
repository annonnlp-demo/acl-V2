from DecisionTree import DTtrain
from knn import knntrain
from lgb import lgbtrain
from svm import svmtrain
from supplement_experiment.data import get_data
from sklearn.model_selection import train_test_split
import sys

sys.path.append('../../.././calibration/')


def train(target_name, split_list, size, feature_list):
    indicator = 38
    if target_name == 'variance':
        indicator = 38
    elif target_name == 'average':
        indicator = 39
    elif target_name == 'indicator3':
        indicator = 40

    data = get_data(size, split_list)
    X, y = data.iloc[:, 3:33], data.iloc[:, indicator]
    X = X[feature_list]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    rmseDT, colDT, ndcgDT, apDT = DTtrain(X_train, X_test, y_train, y_test, target_name)
    rmseKnn, colKnn, ndcgKnn, apKnn = knntrain(X_train, X_test, y_train, y_test, target_name)
    rmselgb, collgb, ndcglgb, aplgb = lgbtrain(X_train, X_test, y_train, y_test, target_name)
    rmsesvm, colsvm, ndcgsvm, apsvm = svmtrain(X_train, X_test, y_train, y_test, target_name)

    return rmseDT, colDT, ndcgDT, apDT, \
           rmseKnn, colKnn, ndcgKnn, apKnn, \
           rmselgb, collgb, ndcglgb, aplgb, \
           rmsesvm, colsvm, ndcgsvm, apsvm


if __name__ == '__main__':
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
    list_features = [feature_name_list_all]
    for indicator in ['variance', 'indicator3']:
        rmseDT_list = []
        colDT_list = []
        ndcgDT_list = []
        apDT_list = []

        rmseKnn_list = []
        colKnn_list = []
        ndcgKnn_list = []
        apKnn_list = []

        rmselgb_list = []
        collgb_list = []
        ndcglgb_list = []
        aplgb_list = []

        rmsesvm_list = []
        colsvm_list = []
        ndcgsvm_list = []
        apsvm_list = []

        for value in list_features:
            rmseDT, colDT, ndcgDT, apDT, \
            rmseKnn, colKnn, ndcgKnn, apKnn, \
            rmselgb, collgb, ndcglgb, aplgb, \
            rmsesvm, colsvm, ndcgsvm, apsvm = train(indicator,
                                                    ['right_grammar', 'basic_word',
                                                     'right_len'],
                                                    987,
                                                    value)
            rmseDT_list.append(rmseDT)
            colDT_list.append(colDT)
            ndcgDT_list.append(ndcgDT)
            apDT_list.append(apDT)

            rmseKnn_list.append(rmseKnn)
            colKnn_list.append(colKnn)
            ndcgKnn_list.append(ndcgKnn)
            apKnn_list.append(apKnn)

            rmselgb_list.append(rmselgb)
            collgb_list.append(collgb)
            ndcglgb_list.append(ndcglgb)
            aplgb_list.append(aplgb)

            rmsesvm_list.append(rmsesvm)
            colsvm_list.append(colsvm)
            ndcgsvm_list.append(ndcgsvm)
            apsvm_list.append(apsvm)

        import pandas as pd

        dataframe = pd.DataFrame({'rmseDT': rmseDT_list,
                                  'colDT': colDT_list,
                                  'ngcdDT': ndcgDT_list,
                                  'apDT': apDT_list,

                                  'rmseKnn': rmseKnn_list,
                                  'colKnn': colKnn_list,
                                  'ngcdKnn': ndcgKnn_list,
                                  'apKnn': apKnn_list,

                                  'rmselgb': rmselgb_list,
                                  'collgb': collgb_list,
                                  'ngcdlgb': ndcglgb_list,
                                  'aplgb': aplgb_list,

                                  'rmsesvm': rmsesvm_list,
                                  'colsvm': colsvm_list,
                                  'ngcdsvm': ndcgsvm_list,
                                  'apsvm': apsvm_list

                                  })

        dataframe.to_excel(f'./data_11_11/{indicator}.xls')
