# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:00:50 2016

@author: admin
"""

import numpy
import theano
import pickle

def load_data(dataset):
    print("Loading data...", flush=True)
    
    # load the data into floats in order to avoid conversions of the 
    # majority of it later

#    train_in = numpy.loadtxt(dataset+"-train.txt", numpy.int32)
#    valid_in = numpy.loadtxt(dataset+"-valid.txt", numpy.int32)
#    test_in = numpy.loadtxt(dataset+"-test.txt", numpy.int32)

# doesn't work with sparse matrices
#    train_in = numpy.load(dataset+"-train.npy")
#    valid_in = numpy.load(dataset+"-valid.npy")
#    test_in = numpy.load(dataset+"-test.npy")

    with open(dataset + '-train', 'rb') as infile:
        nptrain_x, nptrain_y = pickle.load(infile)
    with open(dataset + '-valid', 'rb') as infile:
        npvalid_x, npvalid_y = pickle.load(infile)
    with open(dataset + '-test', 'rb') as infile:
        nptest_x, nptest_y = pickle.load(infile)
    
    print("Shapes: ", nptrain_x.shape, nptrain_y.shape, npvalid_x.shape,
          npvalid_y.shape, nptest_x.shape, nptest_y.shape)
    
    # the y vectors are not very sparse, so convert to regular (dense) arrays...
#    print("Extracting labels...", flush=True)
#    nptrain_x = train_in[:,1:]
#    nptrain_y = train_in[:,0].toarray()
#
#    npvalid_x = valid_in[:,1:]
#    npvalid_y = valid_in[:,0].toarray()
#    
#    nptest_x = test_in[:,1:]
#    nptest_y = test_in[:,0].toarray()

#    theano expects X to use default floating point and Y to be int32
#    # here's code for doing conversions if necessary
#    train_x = theano.shared(numpy.asarray(nptrain_x, dtype=theano.config.floatX),
#                             borrow=True)
#    train_y = theano.shared(numpy.asarray(nptrain_y, dtype=numpy.int32), borrow=True)
#    train_y = theano.shared(nptrain_y, borrow=True)
#    valid_x = theano.shared(numpy.asarray(npvalid_x, dtype=theano.config.floatX),
#                             borrow=True)
#    valid_y = theano.shared(numpy.asarray(npvalid_y, dtype=numpy.int32), borrow=True)
#    test_x = theano.shared(numpy.asarray(nptest_x, dtype=theano.config.floatX),
#                             borrow=True)
#    test_y = theano.shared(numpy.asarray(nptest_y, dtype=numpy.int32), borrow=True)

    print("Creating theano variables...", flush=True)
    
    # the pickled versions should have the right datatypes
    train_x = theano.shared(nptrain_x.toarray(), borrow=True)
    train_y = theano.shared(nptrain_y, borrow=True)
    valid_x = theano.shared(npvalid_x.toarray(), borrow=True)
    valid_y = theano.shared(npvalid_y, borrow=True)
    test_x = theano.shared(nptest_x.toarray(), borrow=True)
    test_y = theano.shared(nptest_y, borrow=True)

    print("Data loaded.", flush=True)
   
    return [(train_x,train_y), (valid_x,valid_y), (test_x, test_y)]
   

if __name__ == '__main__':
    load_data('sdrugint')
