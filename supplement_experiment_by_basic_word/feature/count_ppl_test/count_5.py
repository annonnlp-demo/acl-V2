from count import ppl
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
ppl('dbpedia', 'content',total=4,order=3)