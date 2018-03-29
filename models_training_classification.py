#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:38:16 2018

@author: jingjing
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def cal_accuracy(y, y_pred):
    return np.sum(y_pred == y) / y_pred.shape[0]

def cal_error(y, y_pred):
    return np.mean((y_pred - y) ** 2)

def evaluate_one_run(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    
    # The mean square error
    error = cal_error(y_test, y_pred)
    print("Residual sum of squares from one train: %.2f" % error)
    
    # Explained variance score: 1 is perfect prediction
    accuracy = cal_accuracy(y_test, y_pred)
    print('Mean accuracy from one train: %.2f' % accuracy)

def evaluate_cross_validation(y, y_pred):
    error = cal_error(y, y_pred)
    print("Residual sum of squares from cross validation: %.2f" % error)
    accuracy = cal_accuracy(y, y_pred)
    print('Mean accuracy from cross validation: %.2f' % accuracy)
    
def LR(X_train, X_test, y_train, y_test, eval_one = False, eval_cross = False):
    print('\n Logistic regression')
    
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    
    if eval_one:
        evaluate_one_run(lr, X_test, y_test)
    
    if eval_cross:
        y_pred_cv = cross_val_predict(lr, X, y, cv=10)
        evaluate_cross_validation(y, y_pred_cv)
    
def SVM(X_train, X_test, y_train, y_test, eval_one = False, eval_cross = False):
    print('\n SVM')
    
    svc = SVC()
    svc.fit(X_train, y_train)
    
    if eval_one:
        evaluate_one_run(svc, X_test, y_test)

    if eval_cross:    
        y_pred_cv = cross_val_predict(svc, X, y, cv=10)
        evaluate_cross_validation(y, y_pred_cv)
    
def RF(X_train, X_test, y_train, y_test, eval_one = False, eval_cross = False):
    print('\n Random forest')
    
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    
    if eval_one:
        evaluate_one_run(clf, X_test, y_test)
    
    if eval_cross:
        y_pred_cv = cross_val_predict(clf, X, y, cv=10)
        evaluate_cross_validation(y, y_pred_cv)
    
def GB(X_train, X_test, y_train, y_test, eval_one = False, eval_cross = False):
    print('\n Gradient Boosting Tree')

    clf = GradientBoostingClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)

    if eval_one:
        evaluate_one_run(clf, X_test, y_test)
    
    if eval_cross:
        y_pred_cv = cross_val_predict(clf, X, y, cv=10)
        evaluate_cross_validation(y, y_pred_cv)

if __name__ == '__main__':
    df = pd.read_csv('features.csv')
#    print(df.describe(), df.dtypes)
    
    def label(x):
        if x == 'pickup':
            return 1
        elif x == 'drop':
            return 2
        else:
            return 3
        
    df['label'] = df['action'].apply(lambda x: label(x))
    
    #df = df[df['label'] != 3]
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
#    X = normalize(X)
    
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    y = np.array(df['label'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    LR(X_train, X_test, y_train, y_test, False, True)
    SVM(X_train, X_test, y_train, y_test, False, True)
    RF(X_train, X_test, y_train, y_test, False, True)
    GB(X_train, X_test, y_train, y_test, False, True)

