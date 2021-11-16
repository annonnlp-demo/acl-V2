from count import ppl

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


ppl('ade')
ppl('ag')

ppl('imdb')
ppl('MR')
ppl('rotten')
ppl('fdu', 'content')
ppl('sst1')
ppl('sst2')
ppl('subj')
