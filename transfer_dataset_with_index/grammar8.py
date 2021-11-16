from get_gram import get_split_dataset_gram


for index in range(13,21):
    print(index)
    get_split_dataset_gram('dbpedia',total=50,order=index,text_name='content')