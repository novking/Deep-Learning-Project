# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:08:40 2016

@author: yiz613
"""
import gzip
import pickle

'''
def load(filename):
        with open(filename, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break
items = load("mnist.pkl.gz")

'''

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    
    
