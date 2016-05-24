# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:02:52 2016

@author: yiz613
"""
import LR_EncodedData
import EncodedGenerator_LR_version

import csv


def col_name():
        b = 'best_validation_loss'
        c = 'test_score'
        d = 'running_time'
        return b, c, d
def condition(hidden_units, train, val, testing):
    a = 'hidden_units: '+ str(hidden_units)
    b = 'training_set: ' + str(train)
    c = 'validation_set: ' + str(val)
    d = 'testing_set: ' + str(testing)
    return a, b, c, d
        
        

    
    
with open('test.csv','a') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    a = condition(100,70000,20000,10000)
    writer.writerow(a)
    b = col_name()
    writer.writerow(b)
    csvfile.close   
for i in range(64):
    EncodedGenerator_LR_version.generator(20000,10000,70000,"encodedrugs_100.txt")
    a = LR_EncodedData.sgd_optimization_mnist()
    with open('test.csv','a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(a)
    csvfile.close
