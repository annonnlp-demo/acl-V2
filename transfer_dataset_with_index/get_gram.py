from datasets import load_dataset
from datasets import load_from_disk
from datasets import concatenate_datasets
import os, string, json
import datasets
from datasets.features import Value, ClassLabel


def get_split_dataset_gram(dataset_name,  text_name='text',total=None,order=None):
    dataset = load_from_disk('./results/'+dataset_name+'/')
    start=0
    end=len(dataset)

    if total!=None and order!=None:
        span=len(dataset)//total
        start=(order-1)*span
        end=start+span

        if total==order:
            end=len(dataset)

    dic={'dataset':dataset_name,'grammar':{}}



    is_bad_rule = lambda rule: rule.message == 'Possible spelling mistake found.' and len(rule.replacements) and \
                               rule.replacements[0][0].isupper()
    import language_tool_python

    tool = language_tool_python.LanguageTool('en-US')



    from tqdm import tqdm
    for index in tqdm(range(start,end)):


        value = dataset[index][text_name]

        value_list = value.split(' ')
        length = len(value_list)
        matches = tool.check(value)
        matches = [rule for rule in matches if not is_bad_rule(rule)]

        other = len(matches)

        dic['grammar'][dataset[index]['index_raw']]=other/length

    import json,pathlib
    pathlib.Path('./results/grammar/'+dataset_name+'/').mkdir(parents=True,exist_ok=True)

    json_str = json.dumps(dic, indent=1)
    with open('./results/grammar/'+dataset_name+'/'+str(order)+'.json', 'w') as json_file:
        json_file.write(json_str)


    print('complish-------'+str(order))








