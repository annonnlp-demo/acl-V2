from get_gram import get_split_dataset_gram


for index in range(28,31):
    print(index)
    get_split_dataset_gram('yelp',total=50,order=index)