from get_gram import get_split_dataset_gram


for index in range(43,51):
    print(index)
    get_split_dataset_gram('yelp',total=50,order=index)