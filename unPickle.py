# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 14:39:24 2016

@author: yiz613
"""
import pickle
import numpy as np
import gzip


def convertData( inputDataFile ):
    
    def load(filename):
        with open(filename, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break
    
    
    items = load(inputDataFile)
    
    unpickleList=list(items)
    inputMatrix=list()
    targetResult = list()
    
    for eachSample in unpickleList:
        targetResult.append(eachSample[0])
        inputMatrix.append(eachSample[1:])
        
    lengthOfInput = len(inputMatrix[0])
    
    
    targetResultNDarray = np.array(targetResult, dtype=np.int64)
    
    
    inputMatrixNDarray = np.array(inputMatrix, dtype=np.float64)
    
    
    theanoMatchedInput = (inputMatrixNDarray, targetResultNDarray)
    return (theanoMatchedInput, lengthOfInput)


def mnistCompare (inputFile):
    with gzip.open(inputFile, 'rb') as f:
        a = pickle.load(f, encoding='latin1')
    return a
    
if __name__ == "__main__":
     a,b = convertData('encoded_test_pickle')
     
     c = mnistCompare ('mnist.pkl.gz')
     
     
     
     
     
     
     
     
     
     
     
     
     