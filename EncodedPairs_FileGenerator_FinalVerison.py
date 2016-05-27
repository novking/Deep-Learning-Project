# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 16:03:53 2016

@author: yiz613
"""

import re
import sqlite3
import time
import random
import math
import pickle
import sys


def main():
    start_time = time.time()
    global noFound # for the drug didn't find from RDF according to given drug uris file
    noFound =0
    
    uris_file = "drugtrain_05pl1_uris.txt"
    rdf_file = 'input.nt.fixed6.nt'
    encoded_output = "encodedrugs.txt"
    ratio = 0.1 #  [positive pairs out of TOTAL poistive groups]/[negative pairs out of TOTAL negative groups] 
    '''the ratio is also depend on how many positive and negative pairs in total in orignal RDF file'''
    
    startEnter = input("""Please put following files in the same working directory:
            'drugtrain_05pl1_uris.txt'
            'input.nt.fixed6.nt'
            'encodedrugs.txt'
            
    Hit Enter to start (default)>>>""")
    
    if len(startEnter)<1:
        print ('\nStart to parsing files...', flush = True)
        
    else:
        sys.exit("Error message")
    
    
    
    def LoadDrugName2SQL (uris):
        print('')
        'input drug names from uris file to sql'
        DrugList = list()
        for i in uris:
            a = re.findall ('#(.+)', i)
            try:
                DrugList.append(a[0].lower())
            except:
                print('Something is wrong with this uri:\n', i)
        sqlInput( DrugList )
        
    def sqlInput (DrugList):
        'input the drug name list into sql with a id, which should be the same number is the row number in encoded file for each drug'
         
        for drug in DrugList:
            cur.execute('''INSERT INTO drugNameTable (drug_Name) VALUES ( ? )''',  (drug.lower(), ) )
        conn.commit()
        
    # modified from Kavitha's extractDataFromFile.py code
    def RDF2SQL(rdf_file):
        'parse RDF by Regular Expression then put the drug-drug relation into SQL'
        counter = 0
        for line1 in rdf_file:
            if 'NoInteraction' in line1:
                try:
                    drug_A, drug_B = RDFre(line1)
                    cur.execute('''INSERT INTO MatchTable (relation, subjectID, objectID)  VALUES ( ?, ?, ? )''', ( 0, drug_A, drug_B ) )
                    counter += 1
                except:
                    pass
            elif 'drugDrugInteraction' in line1:
                try:
                    drug_A, drug_B = RDFre(line1)
                    cur.execute('''INSERT INTO MatchTable (relation, subjectID, objectID)  VALUES ( ?, ?, ? )''', ( 1, drug_A, drug_B ) )
                    counter += 1
                except:
                    pass
        conn.commit()
        
    def RDFre(line2):
        'output the ID for durgA and B'
        tri = re.findall(r"[<]([^>]*)[>]", line2)
        drug_AA = re.findall('.+/(.+?)$', tri[0])
        drug_BB = re.findall('.+/(.+?)$', tri[2])
        drugAid = findDrugFromSql ( drug_AA[0].lower() )
        drugBid = findDrugFromSql ( drug_BB[0].lower() )
        return drugAid, drugBid
            
    def findDrugFromSql(drugsName):
        'find the drug id from sql database by drug name'
        global noFound
        
        cur.execute("SELECT id FROM drugNameTable WHERE drug_Name= ?", (str(drugsName),))
        a = cur.fetchone()
        if a is not None:
            b=a[0]
            #print ('found it')
            return b
        else:
            #print ('drug name is not found: ',drugsName)
            noFound += 1
            return None
            
    
    def CombineAndShuffle(ratio = 0.1):
        #output will be three list, each list have pos and neg drug interaction in tuple
        cur.execute('''SELECT * FROM MatchTable WHERE relation = 1''')
        pos_vectors = cur.fetchall()
        
        cur.execute('''SELECT * FROM MatchTable WHERE relation = 0''')
        neg_vectors = cur.fetchall()
        
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
        
        random. shuffle ( train_data)
        random.shuffle(valid_data)
        random.shuffle(test_data)
        
        return train_data, valid_data, test_data, ratio
        
    
    def add_drugs( encoded_file_readlines, tuple_for_drugs_interaction):
        "return a list with first digit as (interaction/non-interaction) and following by drugA and drugB's encoded data"
        interaction_list = list()
        interaction, drugA, drugB = tuple_for_drugs_interaction
        
        try:
            ############## Because the drug name ID in sql start at 1, but in encoded file drug start at 0. 
            ############## so i have to do drugIDinSql-1 ==> to give the right line for the drug
            B = encoded_file_readlines[drugB-1].strip('\n')
            
            A = encoded_file_readlines[drugA-1].strip('\n')
        
          
            AA = list(map(float, A.split()))
            BB = list(map(float, B.split()))
            Int = int(interaction)    
            
            interaction_list.append( Int )
            interaction_list.extend( AA )
            interaction_list.extend( BB )
            	
            #print (interaction_list)
            return interaction_list
        except:
            aaa = 1
            print (drugA, drugB, "doesn't work. Some problem here")
            
            return aaa
    
    def createFile(tupleList, encoded_file, savedFileName, reduce_size=0):
        #create a pickle file
       
        
            
        cAppended = 0
        print('\nGenerating [ %s ] file right now...\n' %savedFileName)
       
        f = open(savedFileName, 'ab')
        for theTuple in tupleList:
            output = add_drugs( encoded_file, theTuple)
            if output == 1: #safety
                continue
            pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
            #p.append(output)
            cAppended += 1
            if (cAppended % 10000) == 0:
                print ('loaded', cAppended,'drug drug relations',flush = True)
            if (reduce_size > 0 and reduce_size<cAppended):
                
                break
                    
                #if cAppended == 50000:
                 #   pickle.dump(p, f, pickle.HIGHEST_PROTOCOL)
             #   p=list()
           # if cAppended == listLen:
           #     pickle.dump(p, f, pickle.HIGHEST_PROTOCOL)
           #     #print('The end')
           #     break
        f.close()
    
    
    
    
    
    ####################################################################################################
    ####################################################################################################
    #########################   Beginning of the codes   ###############################################
    ####################################################################################################
    ####################################################################################################
    
    
    
        
    conn = sqlite3.connect(r'drugNameFile.sqlite')
    cur = conn.cursor()
    print ('Generating SQL file...')
    cur.executescript('''
    DROP TABLE IF EXISTS drugNameTable;
    DROP TABLE IF EXISTS MatchTable;
    
    CREATE TABLE drugNameTable ( id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
        drug_Name   TEXT UNIQUE NOT NULL);
    CREATE TABLE MatchTable ( relation INTEGER NOT NULL,
        subjectID  INTEGER NOT NULL,
        objectID INTEGER NOT NULL)''')
    
    
    StepOne = open(uris_file, 'r')
    LoadDrugName2SQL(StepOne)
    cur.execute('''SELECT MAX(id) FROM drugNameTable''')
    loaded, = cur.fetchall()[0]
    #print ('\n-------- Performace Results ------------\n\nTOTAL [ %d ] unique drug names loaded into SQL Table - drugNameTable' % (loaded))
    conn.commit()
    StepOne.close()
    ###################Step One done
    
    
    StepTwo = open(rdf_file,'r')
    RDF2SQL(StepTwo)
    conn.commit()
    StepTwo.close()
    
    cur.execute('''SELECT COUNT(*) FROM MatchTable''')
    pair, = cur.fetchall()[0]
    #print ('\n[ %d ] pairs of drug-drug relations loaded from RDF to SQL Table - MatchTable: ' % (pair))
    train, val, test, ratio = CombineAndShuffle(ratio)
    ################### Step Two done
    
            
            
    StepThree = open(encoded_output)
    encoded_file = StepThree.readlines()
    
    # mac, unix, linux, and windows all can generate the file within a folder
    
    createFile( val, encoded_file, 'encoded_valid_pickle',10000)
    createFile (test, encoded_file, 'encoded_test_pickle',10000)
    createFile (train, encoded_file, 'encoded_train_pickle',80000)
    
    
    
    
    StepThree.close()
    #####################Step Three done
    
if __name__ == "__main__":
    main()
    
    print ('\n-------- Performace Results ------------\n\nTOTAL [ %d ] unique drug names loaded into SQL Table - drugNameTable' % (loaded))
    print ('\n[ %d ] pairs of drug-drug relations loaded from RDF to SQL Table - MatchTable: ' % (pair))
    cur.execute('''SELECT  relation, count(relation) FROM MatchTable GROUP by relation''' )
    statistics = cur.fetchall()
    for item in statistics:
        rel , times = item
        if rel ==1:
            rel = 'Interacting Pairs'
        else:
            rel = 'Non-acting Pairs'
        print ('%s pairs : [ %d ]' % (rel, times))
    print ( 'Numbers of drugs are not found in drugNameTable : [', noFound, ']')
    print ('\nGenerated Three Files:\nTraining file: [ %d ] pairs\nValid file: [ %d ] pairs\nTest file:[ %d ] pairs\n\nThe ratio for all three files\n[positive pairs out of TOTAL poistive groups]/[negative pairs out of TOTAL negative groups]  is\n [ 1 : %.1f ]  (or %.2f)' %(len(train), len(val), len(test), 1/ratio, ratio))
    conn.close()
    print ("\n------ %s seconds ------" % ( time.time() - start_time) )
    
    """
    def load(filename):
        with open(filename, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

items = load(myfilename)
"""