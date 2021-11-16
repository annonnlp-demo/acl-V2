from datasets import load_from_disk
dataset = load_from_disk('./results/'+'rotten'+'/')

print(dataset[1090]['index_raw'])