def get_true_file_name(path, value):
    import os
    tem_list = os.listdir(path)
    for va in tem_list:
        if value.lower() in va.lower():
            return va

    return None


def compare(list_list):
    print(len(list_list[0]))
    results = []
    for index in range(len(list_list[0])):
        if (list_list[0][index] == list_list[1][index] and list_list[0][index] == list_list[2][index] and
                list_list[0][index] == list_list[3][index] ):
            results.append(index)
    print(len(results))
    return results

def pic(list1,list2):
    results=[]
    for value in list2:
        results.append(list1[value])
    return results

if __name__ == '__main__':

    root = 'colabration/supplement_experiment_hy_test/'
    model_list = ['bert',  'cnn', 'lstm-self', 'lstm']
    #file_name_list = ['yelp','QC','IMDB','dbpedia','CR','atis','ag_news']
    #file_name_list = ['dbpedia']
    file_name_list = ['mr','ade']

    for value in file_name_list:
        print(value)
        sentence1_list = []
        sentence2_list = []
        sentence3_list = []
        sentence4_list = []
        #sentence5_list = []
        true_list = []
        bert_list = []
        cnn_list = []
        #cbow_list = []

        lstm_list = []
        lstm_self_list = []

        for va in model_list:
            real_file_name = get_true_file_name(root + va, value)

            real_file = root + va + '/' + real_file_name

            import pandas as pd

            if va == 'bert':
                names = ['sentence', 'true', 'pred', 'pro','true_or_not']
            #elif va == 'cbow':
                #names = ['sentence', 'true', 'pred']
            elif va == 'cnn':
                names = ['sentence', 'true', 'pred', 'pro', 'right_or_not']
            else:
                names = ['sentence', 'true', 'pred', 'pro','right_or_not']

            df = pd.read_csv(real_file, sep='\t', names=names)

            if va == 'bert':

                sentence1_list = []
                tem=df['sentence'].to_list()
                for value_ in tem:
                    tem_value=value_.strip()

                    sentence1_list.append(tem_value)

                true_list = df['true'].to_list()
                bert_list = df['pred'].to_list()
            #elif va == 'cbow':
                #sentence2_list = []
                #tem=df['sentence'].to_list()
                #for value_ in tem:
                    #tem_value=value_.strip()
                    #sentence2_list.append(tem_value)
                #cbow_list = df['pred'].to_list()
            elif va == 'cnn':
                sentence2_list = []
                tem=df['sentence'].to_list()
                for value_ in tem:
                    tem_value=value_.strip()
                    sentence2_list.append(tem_value)
                cnn_list = df['pred'].to_list()
            elif va == 'lstm-self':
                sentence3_list = []
                tem=df['sentence'].to_list()
                for value_ in tem:
                    tem_value=value_.strip()
                    sentence3_list.append(tem_value)
                lstm_self_list = df['pred'].to_list()
            elif va == 'lstm':
                sentence4_list = []
                tem=df['sentence'].to_list()
                for value_ in tem:
                    tem_value=value_.strip()
                    sentence4_list.append(tem_value)
                lstm_list = df['pred'].to_list()

        results = compare([


            sentence1_list,

            sentence2_list,
            sentence3_list,
            sentence4_list,
        ])

        import pandas as pd

        dataframe = pd.DataFrame({'sentence': pic(sentence1_list,results),

                                  'ground_trouth': pic(true_list,results),
                                  '1': pic(bert_list,results),
                                  '2': pic(cnn_list,results),

                                  '3': pic(lstm_self_list,results),
                                  '4': pic(lstm_list,results),

                                  })

        dataframe.to_csv('./results/' + value + '.tsv', index=False, sep='\t', header=False)
