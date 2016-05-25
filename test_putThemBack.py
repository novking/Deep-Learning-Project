# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:30:28 2016

@author: yiz613
"""
import re
import sqlite3
import os
import random
import math
import pickle
import unPickle

conn = sqlite3.connect(r'matchTest.sqlite')
cur = conn.cursor()
#print ('Generating SQL file...')
cur.executescript('''
DROP TABLE IF EXISTS drugNameTable;
CREATE TABLE drugNameTable ( id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    drug_Name   INTEGER  NOT NULL);
''')
    
def afunc(inputDataFile):
    def load(filename):
        with open(filename, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break
    
    
    items = load(inputDataFile)
    
    unpickleList=list(items)
    
    targetResult = list()
    
    for eachSample in unpickleList:
        targetResult.append(int(eachSample[0]))
        
    return targetResult

  
Druglist =list()
DrugList =  afunc("encoded_train_pickle")

for drug in DrugList:
    cur.execute('''INSERT INTO drugNameTable (drug_Name) VALUES ( ? )''',  (drug, ) )
conn.commit()

DrugList =  afunc("encoded_valid_pickle")

for drug in DrugList:
    cur.execute('''INSERT INTO drugNameTable (drug_Name) VALUES ( ? )''',  (drug, ) )
conn.commit()

DrugList =  afunc("encoded_test_pickle")

for drug in DrugList:
    cur.execute('''INSERT INTO drugNameTable (drug_Name) VALUES ( ? )''',  (drug, ) )
conn.commit()


cur.close()