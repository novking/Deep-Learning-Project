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

tree = ET.ElementTree(file= 'drugbank.xml')
root = tree.getroot()

# for elem in root[3].iter():
#     print (elem)
#     for i in elem:
#         if i.tag=="{http://www.drugbank.ca}half-life":
#             print (i.text)
# for i in range(1,10):
#     print (root[i][3].tag, root[i][3].text )
    # this is the result:
    # {http: // www.drugbank.ca}name
    # Cetuximab
    # {http: // www.drugbank.ca}name
    # Dornase
    # alfa
    # {http: // www.drugbank.ca}name
    # Denileukin
    # diftitox
    # {http: // www.drugbank.ca}name
    # Etanercept
    # {http: // www.drugbank.ca}name
    # Bivalirudin
    # {http: // www.drugbank.ca}name
    # Leuprolide
    # {http: // www.drugbank.ca}name
    # Peginterferon
    # alfa - 2
    # a
    # {http: // www.drugbank.ca}name
    # Alteplase
    # {http: // www.drugbank.ca}name
    # Sermorelin


for elements in root.iter():
    count=0
    for i in elements:
        count+=1
        if i.tag == "{http://www.drugbank.ca}half-life" and i.text:
            content = re.findall(r'(\d+)((?: hours| mins| days| day| hrs| min| h| minutes| hour|\) hour|\) hours|h| Hrs| weeks| week| secs| Hours| seconds|\+ hours|\) minutes| months))', i.text)
            if not content:
                print (i.text)
            for each_tuple_result in content:
                digit = float(each_tuple_result[0])
                unit = each_tuple_result[1]
                if unit == " days" or unit==" day": #be careful, there's a space in front
                    final_result = digit
                elif unit == " mins" or unit==" min" or unit==" minutes" or unit==") minutes":
                    final_result = digit/60.0
                elif unit == " weeks" or unit == " week":
                    final_result = digit*24*7
                elif unit == " secs" or unit == " seconds":
                    final_result = digit/3600
                elif unit == " months":
                    final_result = digit*24*30
                else:
                    final_result = digit*24
                #print (final_result)

            #
            # num = list(map(int, content))
            # if num:
            #     avg = sum(num)/float(len(num))
            #     print (avg)
            # else:
            #     print (i.text)
# import re
# l = "this is cool, 123hr or 5% or 0"
# obj = re.findall(r'(\d+)(?:hr|%)', l)
# print (obj)