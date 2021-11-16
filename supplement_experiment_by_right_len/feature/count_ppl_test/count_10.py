from count import ppl
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
ppl('yelp',total=4,order=4)