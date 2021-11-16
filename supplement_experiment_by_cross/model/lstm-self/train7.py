from util import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'






train_('fdu',2,'content',total=10,order=8)