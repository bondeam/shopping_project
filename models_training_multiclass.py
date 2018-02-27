#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 21:21:11 2018

@author: jingjing
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

if __name__ == '__main__':
    df = pd.read_csv('features.csv')
    print(df.describe(), df.dtypes)
    
    def label(x):
        if x == 'pickup':
            return 1
        elif x == 'drop':
            return 2
        elif x == 'noise':
            return 3
        else:
            return 0
        
    df['label'] = df['action'].apply(lambda x: label(x))
    X = df[['kurtosis', 
            'maximum', 
            'mean', 
            'mean_diff', 
            'median', 
            'minimum', 
            'percentile_25', 
            'percentile_75', 
            'skewness', 
            'standard_dev', 
            'variance', 
            'count']].values
    y = np.array(df['label'])
    