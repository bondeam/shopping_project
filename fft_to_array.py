# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:09:25 2018

@author: satya
"""
import numpy as np
from numpy import genfromtxt

root_file = 'mc_butter_pickNdrop1'
path = './fft_data_sliced/'+root_file+'.csv'
fft_data = genfromtxt(path,delimiter=',')
max_val = max(fft_data)
print(root_file)
print(max_val)
print(np.where(fft_data==max_val))

# Temporary Maximum noise and index log
# mc_butter_drop8 --> 0.0818 @ 224
#
# mc_butter_noise2 --> 0.0208 @ 159
# mc_butter_noise4 --> 0.0134 @ 159
#
# mc_butter_pick6 --> 0.039 @ 222
# 
# mc_butter_pickNdrop1 --> 0.3099 @ 159
# mc_butter_pickNdrop3 --> 0.3199 @ 158
# mc_butter_pickNdrop5 --> 0.0704 @ 131 
# mc_butter_pickNdrop7 --> 0.23 @ 163
# mc_butter_pickNdrop9 --> 0.187 @ 175
# mc_butter_pickNdrop10 --> 0.14 @ 167