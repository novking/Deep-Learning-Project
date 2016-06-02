# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:14:23 2016

@author: admin
"""

# !!! IMPORTANT ISSUES !!!!
# 1) My current drug pred ids do not match those used in the classification
#   file. Currently strip out local ids and convert to lower-case
# 2) Some drugs appear multiple imes in drug_id_file, how to handle?
#   Currently, just adds to the list

import re
import math
import numpy
import random
import time
import theano
import scipy
import pickle

drug_vec_file = 'drugtrain_05pl1.txt'       # pseduo-one hot vectors for drugs
drug_id_file = 'drugtrain_05pl1_uris.txt'   # uris for above file
drug_class_file = 'input.nt.fixed6.nt'      # file with pos/neg examples
# drug_class_file = 'input.posneg.small.nt'      
outfname = 'drugint'
float_type = theano.config.floatX

# modified from Kavitha's extractDataFromFile.py code
def loadClassifications():
    f = open(drug_class_file, 'r')

    pos_pairs = []
    neg_pairs = []

    for line in f:
        if 'NoInteraction' in line:
            p = findPairs(line, 'NoInteraction', neg_pairs)
        if 'drugDrugInteraction' in line:
            p = findPairs(line, 'drugDrugInteraction', pos_pairs)
    
    # print("Pos: ", pos_pairs, "\nNeg: ", neg_pairs)    
    return pos_pairs, neg_pairs

def buildVectors(pos_pairs, neg_pairs, drug_vec):
    # get vectors as (d1,d2, label) tuples
    pos_vectors = getDrugTuples(pos_pairs, 1, drug_vec)
    neg_vectors = getDrugTuples(neg_pairs, 0, drug_vec)
    

    print("Splitting into train, valid and test...", flush=True)
    valid_split = int(math.ceil(len(pos_vectors) * .8))
    test_split = int(math.ceil(len(pos_vectors) * .9))
    
    train_data = pos_vectors[:valid_split]
    valid_data = pos_vectors[valid_split:test_split]
    test_data = pos_vectors[test_split:]
    
    valid_split = int(math.ceil(len(neg_vectors) * .8))
    test_split = int(math.ceil(len(neg_vectors) * .9))
    
    # note extend concatenates, append will put the second list as a single element
    train_data.extend(neg_vectors[:valid_split])
    valid_data.extend(neg_vectors[valid_split:test_split])
    test_data.extend(neg_vectors[test_split:])

    # print("Lengths: ", len(train_data), len(valid_data), len(test_data), flush=True)
    
    # shuffle data
    print("Shuffling the data sets...", flush=True)
    random.shuffle(train_data)
    random.shuffle(valid_data)
    random.shuffle(test_data)

#    for i in range(5):
#        print(train_data[i],", ")
#    print(valid_data)
#    print(test_data)
        
    # convert vectors
    print("Converting to numpy ND arrays...", flush=True)
    # print("Training set...", flush=True)
    train_data_xy  = buildXYArrays(train_data)
    # print("Validation set...", flush=True)
    valid_data_xy = buildXYArrays(valid_data)
    # print("Test set...", flush=True)
    test_data_xy = buildXYArrays(test_data)
    
    return train_data_xy, valid_data_xy, test_data_xy

    
# modified from Kavitha's extractDataFromFile.py code
def findPairs(line, matchStr, pair_list):
    it = re.finditer(r"[<]([^>]*)[>]", line)
    pair = []
    for match in it:
        # print("match.group(): ", match.group(), "; matchStr: ", matchStr)
        if matchStr not in match.group():
            normUri = match.group(1).lower()
            # extract local part
            lastSlash = normUri.rindex("/")
            normUri = normUri[lastSlash+1:]
            pair.append(normUri)
    
    # only enter pairs one way, so we can split data correctly for input processing
    if pair[0] < pair[1]:
        pair_list.append(pair)
        
# given a list of pairs of drugs and a label, looks up the drug information
# and builds a ndarray
def getDrugTuples(pair_list, label, drug_vec):
    # drug_vecs = numpy.empty((0, drug_vec.shape[1]*2),dtype='uint8')

    # avoid excessive memory ops by first collecting pairs of vectors in temp
    # and then building the drug vector one time from this
    temp = []
    count = 0;
    skips = 0;

    for d1,d2 in pair_list:
        d1_vec, found = getDrugDetails(d1, drug_vec)
        if not(found):
            skips+=1
            continue

        d2_vec,found = getDrugDetails(d2, drug_vec)
        # print("d1_vec: ", d1_vec.shape, "d2_vec: ", d2_vec.shape)
        if not(found):
            skips+=1
            continue
            
        # assume each returns a single row
        # pair_vec1 = numpy.append(d1_vec, d2_vec).reshape((1,d1_vec.shape[0]+d2_vec.shape[0]))
        temp.append((d1_vec,d2_vec,label))
        # flip order for symmetry, there is a small probability that an example from
        # the training set has a flip in the validation, but I'm willing to live with that
        # pair_vec2 = numpy.append(d2_vec, d1_vec).reshape((1,d1_vec.shape[0]+d2_vec.shape[0]))
        temp.append((d2_vec,d1_vec,label))
        
        # print("pair_vec: ", pair_vec.shape, "drug_vecs: ", drug_vecs.shape)
        # drug_vecs = numpy.concatenate([drug_vecs,pair_vec1, pair_vec2],axis=0)
        count+=1
        if count % 100000 == 0:
            print("Processing... ", count, "pairs found as of ", time.ctime(), " (", skips, "skips)", flush=True)

    print("Processed ", count, "pairs.")
    if skips > 0: 
        print("Skipped ", skips, " example(s) due to inability to find both drugs!")
    
    return temp
    
def buildXYArrays(vec_tuples):
    drug_width=drug_vec.shape[1]
    # build drugs using Theano's default float type to avoid conversion later...
    # note, for 5 paths of length 1 and 150,000 training examples, this is
    # approximately 7x10^9 elements, i.e. 7GB for 1 byte elements
    
    # drug_vecs = numpy.empty((len(vec_tuples), drug_width*2+1),dtype=theano.config.floatX)
    
    #let try making the matrix sparse
    # note liL_matrix is 200x faster than csr, and coo can't be used this way...
    drug_vec_X = scipy.sparse.lil_matrix((len(vec_tuples), drug_width*2), dtype=float_type)    
    drug_vec_Y = numpy.zeros((len(vec_tuples)), dtype=numpy.int32)
    for i in range(len(vec_tuples)):
        # print(i ," data: ", vec_tuples[i])
        d1_vec, d2_vec, label = vec_tuples[i]
        drug_vec_Y[i] = label
        drug_vec_X[i, 0:drug_width] = d1_vec
        drug_vec_X[i, drug_width:] = d2_vec
        if i % 10000 == 0 and i>0:
            print(i, "rows built at ", time.ctime(), flush=True)
        
    # print(drug_vecs)
    

    # add the classification as the first column
    # labels = numpy.ones((drug_vecs.shape[0],1))*label
    # labels = numpy.ones((drug_vecs.shape[0]))*label
    # drug_vecs[0:drug_vecs.shape[0],0] = labels.reshape((labels.shape[0],1))
    # drug_vecs[:,0] = labels

    # return numpy.concatenate([labels,drug_vecs], axis=1)
    return (scipy.sparse.csr_matrix(drug_vec_X), drug_vec_Y)
    
# returns the vector information for the last occurence of the drug
def getDrugDetails(drug_id, drug_vec):
    if (drug_id in drug_uri2id):
        drug_line = drug_uri2id[drug_id]
        # print("Drug: ", drug_id, "Line: ", drug_line)
        return drug_vec[drug_line,:], True
    else:
        # print("Warning: Couldn't find vector for drug ", drug_id)
        return numpy.zeros((drug_vec.shape[1])), False
    
def sampleMatrix(x, rows):
    if rows > x[0].shape[0]:
        rows = x[0].shape[0]
    for i in range(rows):
        print("Y=", x[1][i], ", X=", x[0][i].toarray())
    
print("Start: ", time.ctime())
# read the drug URI to id mappings
drug_uri2id = {}
with open(drug_id_file,"rt") as idf:
    for line in idf:
        id,uri = line.split()
        # normalize the URI (local name all lower case)
        lastHash = uri.rindex("#")
        drug = uri[lastHash+1:]
        drug = drug.lower()
        drug_uri2id[drug] = int(id)   # store as int to avoid conversion later
# keys = list(drug_uri2id.keys())
# print("Drug URI keys sample: ", keys[0:9])
print("Loaded ", len(drug_uri2id.items()), " drug ids", flush=True)

pos_pairs, neg_pairs = loadClassifications()
print("Loaded ", len(pos_pairs), " positive and ", len(neg_pairs), " negative classifications", flush=True)

# load drug vector information
drug_vec = numpy.loadtxt(drug_vec_file, numpy.int8, "#", ",")
print("Loaded ", drug_vec.shape, " array of drug vectors", flush=True)

train_data, valid_data, test_data = buildVectors(pos_pairs, neg_pairs, drug_vec)

print("Result summary:")
print("Train=", train_data[0].shape, ", Valid=", valid_data[0].shape, ", Test=", test_data[0].shape, flush=True)
print("Train sample")
sampleMatrix(train_data, 5)
print("Valid sample")
sampleMatrix(valid_data, 5)
print("Test sample")
sampleMatrix(test_data, 5)

#numpy.savetxt(outfname + '-train.txt', train_data, fmt='%d')        
#numpy.savetxt(outfname + '-valid.txt', valid_data, fmt='%d')        
#numpy.savetxt(outfname + '-test.txt', test_data, fmt='%d')        

# try to save time and space by automatically Pickling
# but don't know if this will use the best level of pickling....

# this doesn't seem to work with scipy sparse matrices
#numpy.save(outfname + '-train', train_data)        
#numpy.save(outfname + '-valid', valid_data)        
#numpy.save(outfname + '-test', test_data)        

with open(outfname + '-train', 'wb') as outfile:
    pickle.dump(train_data, outfile, pickle.HIGHEST_PROTOCOL)
with open(outfname + '-valid', 'wb') as outfile:
    pickle.dump(valid_data, outfile, pickle.HIGHEST_PROTOCOL)
with open(outfname + '-test', 'wb') as outfile:
    pickle.dump(test_data, outfile, pickle.HIGHEST_PROTOCOL)

print("End: ", time.ctime())
