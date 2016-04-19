# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:59:41 2016

@author: yiz613
"""
import pickle
import numpy as np


    
def load(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break
    

inputDataFile = 'valid_relation'


a = load(inputDataFile)
b = next(a)

targetResultNDarray = np.array(b, dtype=np.int64)
print (targetResultNDarray)