#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 00:50:56 2019

"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def load_data(training_path="dataset/train.csv", testing_path="dataset/test.csv", shuffle_data=False):
    training_dataset = pd.read_csv(training_path)
    test_dataset = pd.read_csv(testing_path)
    if shuffle_data == True:    
        training_dataset = shuffle(training_dataset)
    return(training_dataset.values[:,:training_dataset.shape[1]-1], training_dataset.values[:,-1:], test_dataset.values)
    
def write_output(data, path="submit.csv"):
    result = np.concatenate((np.arange(1,data.shape[0]+1).reshape((data.shape[0],1)),data), axis=1)
    np.savetxt(path, result, delimiter=",", header='ID,Predicted', comments='', fmt='%1d')