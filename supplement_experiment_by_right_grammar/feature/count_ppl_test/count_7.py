from count import ppl
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
ppl('yelp',total=4,order=1)
ppl('yelp',total=4,order=2)