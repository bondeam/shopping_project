#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:38:16 2018

@author: jingjing
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_predict

def LR(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    lr = LinearRegression(normalize=True)
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)

    # The mean square error
    print("Residual sum of squares from one train: %.2f"
          % np.mean((y_pred - y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score from one train: %.2f' % lr.score(X_test, y_test))
    
    y_pred_cv = cross_val_predict(lr, X, y, cv=5)
    # The mean square error
    print("Residual sum of squares from cross validation is: %.2f"
          % np.mean((y_pred_cv - y) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score from cross validation: %.2f' % lr.score(X, y))
    
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
    
    LR(X, y)

