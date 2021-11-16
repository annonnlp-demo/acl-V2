import statistics



import numpy as np
from scipy import stats


indic1_list=[3.967676188,
             4.268952838,
             2.68549654,
             3.321311287,
             2.332743163,
             1.769346466,
             1.424953216,
             0.8425718565,
             0.2168153977]

indic4_list=[
    204.3968227,
    62.1666257,
    48.82904084,
    25.17886087,
    23.18280156,
    13.90308219,
    4.634660334,
    2.90940062,
    0.2128585167,

]

'''
final_list=[
    86.14,
    91.73,
    98.58,
    86.91,
    92.00,
    98.98,

]
'''
'''
final_list=[
    78,
    90.833,
    67.86,
    87.29,
    91.65,
    81.083,
]
'''
'''
final_list=[
    0.91,
    0.92,
    0.87,
    0.78,
    0.81,
    0.68,
]
'''
final_list=[
    0.86266,
    0.909,
    0.859,
    0.919,
    0.867,
    0.7835,
    0.78,
    0.8093,
    0.68,
]

print('spearmanr')
print('indicator4:')
print(stats.spearmanr(indic4_list,final_list))

print('indicator1:')
print(stats.spearmanr(indic1_list,final_list))

print('pearsonr')
print('indicator4:')
print(stats.pearsonr(indic4_list,final_list))

print('indicator1:')
print(stats.pearsonr(indic1_list,final_list))


