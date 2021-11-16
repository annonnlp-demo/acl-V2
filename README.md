# acl-code

This is the code contains all the python scripts to generate all the samples for dataset. And you ca use these script to get the features proposed in our paper for a new dataset to evaluate the value of this new dataset!

### 1.Requirement

First, prepare your own python virtual environment by:

```
python3 -m venv tutorial-env
```

or

```
conda create -n your_env_name python=X.X
```

Then, run the following command to prepare the library needed in this project.

```
pip install -r requirements.txt
```

### 2.Prepare datasets

the dataset we use is in two format: one is in the Dataset Object defined in [Huggingface](https://huggingface.co/docs/datasets/processing.html), the other one is in the .csv format. Specifically:'label,sentence\n'.

### 3.Train the dataset using one of the cnn/bert/lstm/lstmattr 

We implement four scripts to train your own dataset, you just only prepare your own dataset in the right format. And you only change some parametres to train your dataset . 

### 4.Get the feature of a new dataset

If you want to evaluate if your dataset is in good discrimination,  you can use one of the 30 scripts in [./supplemnet_experiment_by_cross/feature/](https://github.com/annonnlp-demo/emnlp-V2/tree/main/supplement_experiment_by_cross/feature). Every dir contains one script to calculate one feature. But you must first save your daataset to Dataset Object like [this](https://huggingface.co/docs/datasets/processing.html) and save your own dataset into your local disk, you can the change the 'root_path'  parameter to the right place.  Then you can get the feature in the './results/' dir.

The following is the feature list:

```
[
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
```

### 5.split your dataset

You may want to take a deep insight into how one feature defined in our paper influences the quality of your dataset, you can use one of the function defined above to split your dataset .You can find an example in the dir [./supplement_experiment_by_right_len/dataset/split.py](https://github.com/annonnlp-demo/emnlp-V2/blob/main/supplement_experiment_by_right_len/dataset/split.py) or [./supplement_experiment_by_basic_word/dataset/split.py](https://github.com/annonnlp-demo/emnlp-V2/blob/main/supplement_experiment_by_basic_word/dataset/split.sh) these datasets will saved in the format of Dataset Object. You can use the [./supplement_experiment_by_basic_word/dataset/transfer_dataset_to_csv.py](https://github.com/annonnlp-demo/emnlp-V2/blob/main/supplement_experiment_by_basic_word/dataset/transfer_dataset_to_csv.py) to transer the dataset to .csv format to satisify the demands of other 3 model.

### 6.extracted the feature for instances in the new generated sub dataset
once the datasets are splited to subsets according to differnet feature, you will get new datasets. with these datasets, you can calculate their features. from these features, you can judge which feature is most important. You can also according to the feature to judge the quality of a dataset.

you can find a example in [acl-V2/supplement_experiment/new/](https://github.com/annonnlp-demo/acl-V2/tree/main/supplement_experiment/new)
