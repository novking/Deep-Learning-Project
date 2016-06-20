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
import json

def half_life(first_level):
    content = []
    if first_level.text:
        content = re.findall(r'(\d+)((?: hours| mins| days| day| hrs| min| h| minutes| hour|\) hour|\) hours|h| Hrs| weeks| week| secs| Hours| seconds|\+ hours|\) minutes| months))', first_level.text)
    if not content:
        return
    for each_tuple_result in content:
        digit = float(each_tuple_result[0])
        unit = each_tuple_result[1]
        if unit == " days" or unit == " day":  # be careful, there's a space in front
            final_result = digit
        elif unit == " mins" or unit == " min" or unit == " minutes" or unit == ") minutes":
            final_result = digit / 60.0
        elif unit == " weeks" or unit == " week":
            final_result = digit * 24 * 7
        elif unit == " secs" or unit == " seconds":
            final_result = digit / 3600
        elif unit == " months":
            final_result = digit * 24 * 30
        else:
            final_result = digit * 24
    return final_result
    
def level(first_level):
    result = []
    for child in first_level:
        for level in child:
            atc = level.text
            if atc in d:
                result.append(d[atc])
    return result


def experimental(first_level):
    result = []
    for ep in first_level: # experimental-properties (ep)
        if ep[0].text=="Water Solubility":
            ep_content = re.findall(r'(\d.+)((?: mg/mL| mg/L))', ep[1].text)
            if not ep_content:
                continue
            try:
                if ep_content[0][1]==" mg/L":
                    result.append(float(ep_content[0][0])/1000)
                else:
                    result.append(float(ep_content[0][0]))
            except:
                pass
        elif ep[0].text=="logP":
            result.append(ep[1].text)
        elif ep[0].text =="logS":
            result.append(ep[1].text)
        elif ep[0].text =="pKa":
            result.append(ep[1].text)
        elif ep[0].text == "Isoelectric Point":
            result.append(ep[1].text)
        elif ep[0].text == "Molecular Weight":
            pass
        elif ep[0].text == "Melting Point":
            pass
        else:
            pass
    return result


with open("output.txt") as f:
    atc_dict= f.read()
d = json.loads(atc_dict)
count = 0
flag = 0
for event, drug in ET.iterparse('drugbank.xml'):
    if drug.tag != '{http://www.drugbank.ca}drug':
        continue
    if not drug.get('type'):
        continue
   
    result = []
    if drug.get('type')=='small molecule':
        result.append(1)
    else:
        result.append(0)
        
    for first_level in drug:
        h = 0
        l=0
        e=0
        if first_level.tag == "{http://www.drugbank.ca}half-life":
            h = half_life(first_level)
       
        if first_level.tag == '{http://www.drugbank.ca}atc-codes':
            l = level(first_level)
            
        if first_level.tag == '{http://www.drugbank.ca}experimental-properties':
            e = experimental(first_level)
            
        if h:
            result.append(h)
        if l:
            result.append(l)
        if e:
            result.append(e)
    if len(result)>=3:
        count+=1
print (count)
    

       