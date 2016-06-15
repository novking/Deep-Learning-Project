# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 16:41:17 2016

@author: yiz613
"""
import re
try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET

# tree = ET.ElementTree(file='drugbank.xml')
# root = tree.getroot()
# count = 0
# for ii in range(3, 19):
#     print(root[ii].attrib)
    # count = 0
    # for elem in root[ii].iter():
    #
    #     count += 1
    #     if elem.attrib:
    #         print(elem.attrib)
    #         print(count)
    #     if elem.tag=='{http://www.drugbank.ca}atc-codes':
    #         for i in elem:
    #             print(i.text)
    #     if elem.tag == '{http://www.drugbank.ca}atc-code':
    #         for j in elem:
    #             print(j.text)
#
#     print()
# for i in range(1,10):
#     print (root[i][3].tag, root[i][3].text )



for whatever in range(3000):
    event, elem = next(ET.iterparse('drugbank.xml'))

    if elem.tag == '{http://www.drugbank.ca}drug':
        flag = 0
        for i in range(len(elem)):
            if elem[i].tag== '{http://www.drugbank.ca}calculated-properties':
                print (i)
    else:
        continue
