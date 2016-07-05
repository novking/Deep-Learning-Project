# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:35:14 2016

@author: yiz613
"""

import numpy as np
import new_unPickle
import numpy



# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

test_set, lengthOfInput_test_set = new_unPickle.convertData('train', 'train_relation')
a, b = test_set
# input dataset
X = a
    
# output dataset            
y = np.array([b.tolist()]).T

# seed random numbers to make calculation
#np.random.seed(1)
# initialize weights randomly with mean 0
syn0 = 2*np.random.random((768,1)) - 1

for iter in range(1000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print ("Output After Training:")
hmm = l1.tolist()
for xxx in hmm:
    if (xxx[0]>0.88):
        print (xxx)
import csv
with open('test.csv', 'w', newline='') as fp:
    a = csv.writer(fp, delimiter=',')
    data = hmm
    a.writerows(data)