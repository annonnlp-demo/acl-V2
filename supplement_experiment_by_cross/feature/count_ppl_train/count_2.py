from count import ppl

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

ppl('fdu', 'content')
