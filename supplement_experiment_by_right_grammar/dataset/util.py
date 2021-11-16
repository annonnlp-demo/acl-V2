def get_split_dataset_name(path):
    from os import listdir
    list = listdir(path)
    return list


def save_json(object, path):
    import json
    json_str = json.dumps(object, indent=1)
    with open(path, 'w') as json_file:
        json_file.write(json_str)


def get_json(path):
    import json
    j = open(path, 'r')
    python_list = json.load(j)
    return python_list


def get_dataset(path,split='train'):
    from datasets import load_from_disk
    from datasets import concatenate_datasets

    dataset = load_from_disk(path+split+'/')

    return dataset

