# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 14:39:24 2016

@author: yiz613
"""
import pickle
import numpy as np

def convertDate( inputDataFile ):
    
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
    
    targetResultNDarray = np.array(targetResult)
    
    inputMatrixNDarray = np.array(inputMatrix)
    
    
    theanoMatchedInput = (inputMatrixNDarray, targetResultNDarray)
    
    print ("""it should be:
        x , y = valid_set
        x = (10000, 784)
        y = (10000,)
        """)
        
    print (" Now this is the the really result:\nx =" + inputMatrixNDarray.shape
    + "\ny = " + targetResultNDarray.shape)

    return theanoMatchedInput