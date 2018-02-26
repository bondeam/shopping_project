# -*- coding: utf-8 -*-
"""
               Feature Extraction
===================================================

This modules take a PATH as input, look all the files
inside, read their content and generate a vector
of statistical feature for each of these files.

The vectors will be persisted to a SQLITE database.

"""

# Author: Caleb De La Cruz P. <cdelacru>

import os
import sys
import re
import pandas as pd

# -*- Util Functions -*-

def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    
    reference: https://stackoverflow.com/a/19308592/1354478
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


# -*- Feature Extraction Functions -*-

def feature_extraction(filepath):
    pattern = r'(.*)data_(.*)_shelf_(.*)_(noise|drop|pickup)\d+\.csv'

    filename = re.match(pattern, filepath)
    if filename is not None:
        features = {}
        features['shelf_type'] = filename.group(2)
        features['item'] = filename.group(3)
        features['action'] = filename.group(4)
        
        df = pd.read_csv(filepath, header=None, names=["time", "signal"])
        stats = df.signal.describe()

        features['mean'] = stats['mean']
        features['standard_dev'] = stats['std']
        features['minimum'] = stats['min']
        features['count'] = stats['count']
        features['percentile_25'] = stats['25%']
        features['median'] = stats['50%']
        features['percentile_75'] = stats['75%']
        features['maximum'] = stats['max']
        
        features['kurtosis'] = df.signal.kurt()
        features['skewness'] = df.signal.skew()
        features['variance'] = df.signal.var()
        features['skewness'] = df.signal.skew()
        features['mean_diff'] = df.signal.diff().mean()
        
        return features
    else:
        print('Skipped invalid file: %s' % filepath)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        print('Please specify the path of the data as a parameter')
        exit(-1)

    full_file_paths = get_filepaths(path)
    features = []
    for filepath in full_file_paths:
        feature = feature_extraction(filepath) 
        if feature is not None:
            features.append(feature)
    df = pd.DataFrame(features)

    df.to_csv('features.csv', index=False, columns=['shelf_type',
        'item',
        'kurtosis',
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
        'action'])



