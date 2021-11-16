kind = ['apparel', 'baby', 'books', 'camera_photo', 'dvd', 'electronics',
        'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines', 'music', 'software',
        'sports_outdoors', 'toys_games', 'video']


def get_object(filename):
    import pickle
    pickle_file = open(filename, 'rb')
    array = pickle.load(pickle_file)
    pickle_file.close()
    return array



def get_two_list():
    results = []

    for i in range(len(kind)):
        for index in range(i + 1, len(kind)):
            results.append([kind[i], kind[index]])

    return results


def get_new_dataset():
    results = get_two_list()

    for value in results:
        dataset1 = value[0]
        dataset2 = value[1]

        dataset_1_train = get_object(
            'colabration/fine_tuning/fdu-mtl/' + dataset1 + '/' + dataset1 + '-train_dataset.pkl')
        dataset_1_test = get_object(
            'colabration/fine_tuning/fdu-mtl/' + dataset1 + '/' + dataset1 + '-test_dataset.pkl')
        dataset_1_dev = get_object(
            'colabration/fine_tuning/fdu-mtl/' + dataset1 + '/' + dataset1 + '-dev_dataset.pkl')
        dataset_2_train = get_object(
            'colabration/fine_tuning/fdu-mtl/' + dataset2 + '/' + dataset2 + '-train_dataset.pkl')
        dataset_2_test = get_object(
            'colabration/fine_tuning/fdu-mtl/' + dataset2 + '/' + dataset2 + '-test_dataset.pkl')
        dataset_2_dev = get_object(
            'colabration/fine_tuning/fdu-mtl/' + dataset2 + '/' + dataset2 + '-dev_dataset.pkl')

        import pathlib
        path='./results/fdu/' + dataset1 + '-' + dataset2 + '/'
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        dataset_1_train.save_to_disk(path + 'train/')
        dataset_1_dev.save_to_disk(path + 'validation/')
        dataset_2_test.save_to_disk(path + 'test/')

        path='./results/fdu/' + dataset2 + '-' + dataset1 + '/'
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        dataset_2_train.save_to_disk(path + 'train/')
        dataset_2_dev.save_to_disk(path + 'validation/')
        dataset_1_test.save_to_disk(path + 'test/')



print(len(get_two_list()))