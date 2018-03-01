#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:38:16 2018

@author: jingjing
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_predict
from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier

def LR(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    lr = LinearRegression(normalize=True)
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)

    # The mean square error
    print("Residual sum of squares from one train: %.2f"
          % np.mean((y_pred - y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Mean accuracy from one train: %.2f' % lr.score(X_test, y_test))
    
    y_pred_cv = cross_val_predict(lr, X, y, cv=10)
    print(y_pred_cv)
    # The mean square error
    print("Residual sum of squares from cross validation is: %.2f"
          % np.mean((y_pred_cv - y) ** 2))
    # Explained variance score: 1 is perfect prediction
    accuracy = accuracy_score(y, y_pred_cv, normalize=False)
    print('Mean accuracy from cross validation: %.2f' % accuracy)
    
def SVM(X, y):
    svc = SVC()
    svc.fit(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    
    # The mean square error
    print("Residual sum of squares from one train: %.2f"
          % np.mean((y_pred - y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Mean accuracy from one train: %.2f' % svc.score(X_test, y_test))
    
    y_pred_cv = cross_val_predict(svc, X, y, cv=10)
    print(y_pred_cv)
    # The mean square error
    print("Residual sum of squares from cross validation is: %.2f"
          % np.mean((y_pred_cv - y) ** 2))
    # Explained variance score: 1 is perfect prediction
    accuracy = accuracy_score(y, y_pred_cv, normalize=False)
    print('Mean accuracy from cross validation: %.2f' % accuracy)
    
def RF(X, y):
    pass
    
def GB(X, y):
    pass

if __name__ == '__main__':
    df = pd.read_csv('features.csv')
    print(df.describe(), df.dtypes)
    
    def label(x):
        if x == 'pickup':
            return 1
        elif x == 'drop':
            return 2
        else:
            return 3
        
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
    
#    LR(X, y)
    SVM(X, y)
#    RF(X, y)
#    GB(X, y)

