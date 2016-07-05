# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:24:02 2016

@author: yiz613
"""
import re
import random
import math
import pickle

def rdf_negative_pairs_generator(rdf_file):
    with open(rdf_file, 'r') as rdf:
        for line in rdf:
            if 'NoInteraction' in line:
                try: 
                    drug_A, drug_B = rdf_parse(line)
                    yield (0, drug_A, drug_B)
                except:
                    print ("warming: something wrong with this:\n", line)
                   
def rdf_positive_pairs_generator(rdf_file):
    with open(rdf_file, 'r') as rdf:
        for line in rdf:
            if 'drugDrugInteraction' in line:
                try:
                    drug_A, drug_B = rdf_parse(line)
                    yield (1, drug_A, drug_B)
                except:
                    print ("warming: something wrong with this:\n", line)
                
def rdf_parse(rdf_line):
    tri = re.findall(r"[<]([^>]*)[>]", rdf_line)
    drug_AA = re.findall('.+/(.+?)$', tri[0])[0].lower()
    drug_BB = re.findall('.+/(.+?)$', tri[2])[0].lower()
    return drug_AA, drug_BB
    
def create_dict_for_each_drug(uris_file):
    drugDict = {}
    with open(uris_file, 'r') as uris:
        for i in uris:
            theDrugName = re.findall ('#(.+)', i)
            theLine = re.findall(r'^([0-9]+)', i)
            try:
                key = theDrugName[0].lower()
                value = int(theLine[0])
                drugDict[key] = value
            except:
                print('Something is wrong with this uri:\n', i)
    print ('create the drugDict')
    #print (drugDict)
    return drugDict

def random_shuffle_combine(pos_vectors, neg_vectors, ratio = 0.3):
    random.shuffle(pos_vectors)
    random.shuffle(neg_vectors)
    
    valid_split = int(math.ceil(len(pos_vectors) * (1-2*ratio)))
    test_split = int(math.ceil(len(pos_vectors) * (1-ratio)))
    
    train_data = pos_vectors[:valid_split]
    valid_data = pos_vectors[valid_split:test_split]
    test_data = pos_vectors[test_split:]
    
    valid_split = int(math.ceil(len(neg_vectors) * (1-2*ratio)))
    test_split = int(math.ceil(len(neg_vectors) * (1-ratio)))
    
    # note extend concatenates, append will put the second list as a single element
    train_data.extend(neg_vectors[:valid_split])
    valid_data.extend(neg_vectors[valid_split:test_split])
    test_data.extend(neg_vectors[test_split:])

    # print("Lengths: ", len(train_data), len(valid_data), len(test_data), flush=True)
    
    # shuffle data
    print("\nShuffling the data sets...", flush= True )
    
    random.shuffle( train_data )
    random.shuffle( valid_data )
    random.shuffle( test_data )
    
    print ("done shuffling")
    return train_data, valid_data, test_data

#def match_drug_encoded_data_pairs(tupleList, encoded_data_file, drugDict, savedFileName):
#    'use the dict and listOfTuple generated by "random_shuffle_combine"'
#    cApp = 0
#    s = open (savedFileName, 'ab')
#    with open(encoded_data_file, 'r') as f:
#        encoded_data_readline = f.readlines()
#        for oneTuple in tupleList:
#            try:
#                relation_int, drugA, drugB = oneTuple
#                relation_str = str(relation_int)+' '
#                drugA_value_str = encoded_data_readline[drugDict[drugA]].strip('\n')
#                drugB_value_str = encoded_data_readline[drugDict[drugB]].strip('\n')
#                final_string = ''.join([relation_str, drugA_value_str,' ', drugB_value_str])
##                print (final_string)
#                pickle.dump(final_string, s ,pickle.HIGHEST_PROTOCOL)
#                cApp += 1
#                if (cApp % 20000 ==1):
#                    print (20000)
#                
#            except:
#                pass
#    print (cApp)
#    s.close()
#    return 0

def match_drug_encoded_data_pairs(tupleList, encoded_data_file, drugDict, savedFileName_matrix, savedFileName_relation, breakpoint):
    'use the dict and listOfTuple generated by "random_shuffle_combine"'
    cApp = 0
    relationList = []
    s = open (savedFileName_matrix, 'ab')
    s_relation = open(savedFileName_relation, 'ab')
    with open(encoded_data_file, 'r') as f:
        encoded_data_readline = f.readlines()
        for oneTuple in tupleList:
            try:
                relation_int, drugA, drugB = oneTuple
                drugA_value_str = encoded_data_readline[drugDict[drugA]].strip('\n')
                drugB_value_str = encoded_data_readline[drugDict[drugB]].strip('\n')
                
                drugA_list = drugA_value_str.split()
                drugB_list = drugB_value_str.split()
                finalA = list(map(float, drugA_list))
                finalB = list(map(float, drugB_list))
                finalA.extend(finalB)
                
                pickle.dump(finalA, s ,pickle.HIGHEST_PROTOCOL)
                relationList.extend([relation_int])
                cApp += 1
                if (cApp % 20000 ==1):
                    print (cApp)
                    
##########################flag##############################################
                if (cApp == breakpoint):
                    break
##########################flag##############################################

            except:
                pass
    print (cApp)
    pickle.dump(relationList, s_relation, pickle.HIGHEST_PROTOCOL)
    s.close()
    s_relation.close()
    return 0
    
def main(encoded_data_file, rdf_file, uris_file):
    pos = rdf_positive_pairs_generator(rdf_file)
    neg = rdf_negative_pairs_generator(rdf_file)
    pos_vectors = [i for i in pos]
    neg_vectors = [j for j in neg]
    train_data, valid_data, test_data = random_shuffle_combine(pos_vectors, neg_vectors, ratio = 0.1)
    savedFileName1 = 'test'
    
    tupleList1 = test_data
    savedFileName2 = 'valid'
    
    tupleList2 = valid_data
    savedFileName3 = 'train'

    tupleList3 = train_data
    drugDict = create_dict_for_each_drug(uris_file)
    
    savedFileName_for_relation1 = 'test_relation'
    savedFileName_for_relation2 = 'valid_relation'
    savedFileName_for_relation3 = 'train_relation'
    
    match_drug_encoded_data_pairs(tupleList1, encoded_data_file, drugDict, savedFileName1, savedFileName_for_relation1, 10000)
    match_drug_encoded_data_pairs(tupleList2, encoded_data_file, drugDict, savedFileName2, savedFileName_for_relation2, 10000)
    match_drug_encoded_data_pairs(tupleList3, encoded_data_file, drugDict, savedFileName3, savedFileName_for_relation3, 20000)


#    match_drug_encoded_data_pairs(tupleList1, encoded_data_file, drugDict, savedFileName1 )
#    match_drug_encoded_data_pairs(tupleList2, encoded_data_file, drugDict, savedFileName2)
#    match_drug_encoded_data_pairs(tupleList3, encoded_data_file, drugDict, savedFileName3)

if __name__ == "__main__":
    encoded_data_file = 'encodedrugs.txt'
    rdf_file = 'input.nt.fixed6.nt'
    uris_file = 'drugNameInOrder.txt'
    main(encoded_data_file, rdf_file, uris_file)
    #one can adjust the input pairs amount inside the main()


    





#data_x = np.loadtxt(r'C:/Users/yiz613/Desktop/logistic_regression/encodedrugs_100.txt', np.float32)