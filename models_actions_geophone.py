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
from sklearn.model_selection import GridSearchCV

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
    
def LR(X_train, X_test, y_train, y_test, eval_one = False, eval_cross = False, gridSearch = False):
    print('\n Logistic regression')
    
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    
    if eval_one:
        evaluate_one_run(lr, X_test, y_test)
    
    if gridSearch:
        param_grid = {
                'penalty': ['l1', 'l2'],
                'C': [0.1, 0.5, 1, 10, 100]
                }
        clf = GridSearchCV(lr, param_grid)
        clf.fit(X, y)
        print(clf.best_params_)
        print(clf.best_estimator_)
        lr = clf.best_estimator_
        
    if eval_cross:
        y_pred_cv = cross_val_predict(lr, X, y, cv=10)
        evaluate_cross_validation(y, y_pred_cv)
        

    
def SVM(X_train, X_test, y_train, y_test, eval_one = False, eval_cross = False, gridSearch = False):
    print('\n SVM')
    
    svc = SVC()
    print(svc)
    svc.fit(X_train, y_train)
    
    if eval_one:
        evaluate_one_run(svc, X_test, y_test)

    if gridSearch:
        param_grid = [
                {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
                ]
        clf = GridSearchCV(svc, param_grid)
        clf.fit(X, y)
        svc = clf.best_estimator_
        print(svc)
        
    if eval_cross:    
        y_pred_cv = cross_val_predict(svc, X, y, cv=10)
        evaluate_cross_validation(y, y_pred_cv)
    

    
def RF(X_train, X_test, y_train, y_test, eval_one = False, eval_cross = False, gridSearch = False):
    print('\n Random forest')
    
    rf = RandomForestClassifier(max_depth=6,min_samples_leaf=6,n_estimators=100)
    rf.fit(X, y)
    print(rf.feature_importances_)
    visFeatures(rf.feature_importances_)
    if eval_one:
        evaluate_one_run(rf, X_test, y_test)
    
    if gridSearch:
        param_grid = {
                'n_estimators': [100, 200, 300, 400, 500],
                'criterion': ['gini', 'entropy'],
                "max_depth" : [1, 2, 3, 4, 5, 6, 7, 8],
                "min_samples_leaf" : [1, 2, 4, 6, 8, 10]
                }
        clf = GridSearchCV(rf, param_grid)
        clf.fit(X, y)
        rf = clf.best_estimator_
        
    if eval_cross:
        print(rf)
        y_pred_cv = cross_val_predict(rf, X, y, cv=10)
        evaluate_cross_validation(y, y_pred_cv)
    
def GB(X_train, X_test, y_train, y_test, eval_one = False, eval_cross = False, gridSearch = False):
    print('\n Gradient Boosting Tree')

    gbdt = GradientBoostingClassifier(learning_rate=0.01,max_depth=2,min_samples_leaf=2,n_estimators=200)
    gbdt.fit(X, y)
    print(gbdt.feature_importances_)
    visFeatures(gbdt.feature_importances_)
    if eval_one:
        evaluate_one_run(gbdt, X_test, y_test)
    
    if gridSearch:
        param_grid = {
                'learning_rate': [0.01, 0.05, 0.1, 0.15],
                'n_estimators': [50, 100, 200],
                "max_depth" : [1, 2, 3, 4, 5, 6, 7, 8],
                "min_samples_leaf" : [1, 2, 4, 6, 8, 10]
                }
        clf = GridSearchCV(gbdt, param_grid)
        clf.fit(X, y)
        gbdt = clf.best_estimator_
        
    if eval_cross:
        print(gbdt)
        y_pred_cv = cross_val_predict(gbdt, X, y, cv=10)
        evaluate_cross_validation(y, y_pred_cv)


def visFeatures(importances):
    from collections import OrderedDict
    import matplotlib.pyplot as plt
    
    featureNames = ['kurtosis', 
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
            'count',
            'centroid',
            'crest',
            'fft_kurtosis',
            'fft_mean',
            'fft_skewness',
            'fft_variance',
            'flatness',
            'rolloff',
            'spread']
    d = {}
    for i in range(len(importances)):
        if importances[i] > 0.01:
            d[featureNames[i]] = importances[i]
    
    sorted_feature_importances = OrderedDict(sorted(d.items(), key=lambda x:x[1], reverse=True))
    D = sorted_feature_importances
    plt.figure()
    plt.bar(range(len(D)), D.values(), align='center')
    plt.xticks(range(len(D)), D.keys(),rotation=90)
    
if __name__ == '__main__':
    df = pd.read_csv('features.csv')
#    print(df.describe(), df.dtypes)
    print(df.columns)
    
    def label(x):
        if x == 'pickup':
            return 1
        elif x == 'drop':
            return 2
        else:
            return 3
        
    df['label'] = df['action'].apply(lambda x: label(x))
    
    
    print(len(df[df['label'] == 1]))
    print(len(df[df['label'] == 2]))
    print(len(df[df['label'] == 3]))
    
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
            'count',
            'centroid',
            'crest',
            'fft_kurtosis',
            'fft_mean',
            'fft_skewness',
            'fft_variance',
            'flatness',
            'rolloff',
            'spread']].values

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    y = np.array(df['label'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
#    LR(X_train, X_test, y_train, y_test, False, True, True)
#    SVM(X_train, X_test, y_train, y_test, False, True, True)
    RF(X_train, X_test, y_train, y_test, False, True, False)
    GB(X_train, X_test, y_train, y_test, False, True, False)

