# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 16:41:17 2016

@author: yiz613
"""
import re, pickle
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
            final_result = digit*24
        elif unit == " mins" or unit == " min" or unit == " minutes" or unit == ") minutes":
            final_result = digit / 60.0
        elif unit == " weeks" or unit == " week":
            final_result = digit * 24 * 7
        elif unit == " secs" or unit == " seconds":
            final_result = digit / 3600
        elif unit == " months":
            final_result = digit * 24 * 30
        else:
            final_result = digit 
    return [final_result/100]
    
def level(first_level):
    result = []
    for child in first_level:
        for level in child:
            atc = level.text
            if atc in d:
                result.append(d[atc]/374.0)
    return result


def experimental(first_level):
    result = [0.12, 0.3 ]
    for ep in first_level: # experimental-properties (ep)
#        if ep[0].text=="Water Solubility":
#            ep_content = re.findall(r'(\d.+)((?: mg/mL| mg/L))', ep[1].text)
#            if not ep_content:
#                continue
#            try:
#                if ep_content[0][1]==" mg/L":
#                    result[0] = (float(ep_content[0][0])/1000)/100
#                else:
#                    result[0] = (float(ep_content[0][0]))/100
#            except:
#                pass
        if ep[0].text=="logP":
            result[0] = (float(ep[1].text)+4)/14
#        elif ep[0].text =="logS" and result[2]!=-1:
#            s = float(ep[1].text)
#            result[2] = (s+10.0)/12.0
#        elif ep[0].text =="pKa":
#            try:
#                result[1] = float(ep[1].text)/14.0
#            except:
#                pass
##        elif ep[0].text == "Isoelectric Point":
##            result[4] = (float(ep[1].text)-1)/11.0
#        elif ep[0].text == "Molecular Weight":
#            try:
#                result[2] =float(ep[1].text)
#            except:
#                pass
        elif ep[0].text == "Melting Point":
            s = re.findall(r'(\d*\.?\d*)(?:°C| °C| dec °C)', ep[1].text)
            try:
                result[1] = float(s[0])/322
            except:
#                print(ep[1].text)
                pass
        else:
            pass
    if sum(result)!=0:
        return result
    else:
        return

def protien_binding(first_level):
    if first_level.text:
        #content = re.findall(r'(\d.+)(?: %|%)', first_level.text)
        #content1 = re.findall(r'(\d+(\.\d{1,2})?)(?: %|%)', first_level.text)
        content2 = re.findall(r'(\d*\.?\d*)(?: %|%)', first_level.text)
        if not content2:
            return
        try:
            x = [i for i in map(float, content2)]
            res = sum(x)/len(x)/100
            return [res]
        except:
            return 0
    return 0
#    for each_tuple_result in content:
#        digit = float(each_tuple_result[0])

def calculated_properties(first_level):
    result = [0.11, 0.01, 0, 0.99, 0.5]
    for ep in first_level: # experimental-properties (ep)
#        if ep[0].text=="Water Solubility" and result[0]==-1:
#            ep_content = re.findall(r'(\d.+)((?: mg/mL| mg/L))', ep[1].text)
#            if not ep_content:
#                continue
#            try:
#                if ep_content[0][1]==" mg/L":
#                    result[0] = (float(ep_content[0][0])/1000)/100
#                else:
#                    result[0] = (float(ep_content[0][0]))/100
#            except:
#                pass
        if ep[0].text=="logP" and result[0]==-1:
            result[0] = (float(ep[1].text)+4)/14
#        elif ep[0].text =="logS" and result[2]!=-1:
#            s = float(ep[1].text)
#            result[2] = (s+10.0)/12.0
        if ep[0].text =="Polarizability" and result[1]==-1:
            polar = re.findall(r'(\d*\.?\d*)', ep[1].text)
            if not polar:
                pass
            else:
                result[1] = (float(polar[0]))/700.0
#        elif ep[0].text == "Isoelectric Point" and result[4]==-1:
#            result[4] = (float(ep[1].text)-1)/11.0
#        elif ep[0].text == "Molecular Weight":
#            pass
#        elif ep[0].text == "Melting Point":
#            pass
#        elif ep[0].text == "InChI":
#            pass
#        elif ep[0].text == "Molecular Formula":
#            pass
#        elif ep[0].text == "Polar Surface Area (PSA)":
#            pass
#        elif ep[0].text == "H Bond Acceptor Count":
#            pass
#        elif ep[0].text == "H Bond Donor Count":
#            pass
        if ep[0].text == "Number of Rings" and result[2]==-1:
            result[2] = (int(ep[1].text))/11
#        if ep[0].text == "SMILES":
#            pass
        if ep[0].text == "pKa (strongest acidic)":
            result[3] = (float(ep[1].text))/7.0
            
        if ep[0].text == "pKa (strongest basic)" and result[4]==-1:
            result[4] = (float(ep[1].text))/14.0
         
#        elif ep[0].text == "Monoisotopic Weight":
#            pass
#        else:
#            pass
  
    return result
      
        
    

with open("json_dict.txt") as f:
    atc_dict= f.read()
d = json.loads(atc_dict)
count = 0
flag = 0
output_file = open('output', 'ab')
for event, drug in ET.iterparse('drugbank.xml'):
    if drug.tag != '{http://www.drugbank.ca}drug':
        continue
    if not drug.get('type'):
        continue
   
    result = [0]*5
    result.append([0.11, 0.01, 0, 0.99, 0.5])
    if drug.get('type')=='small molecule':
        result[0] = [1]
    else:
        result[0] = [0]
    items = 0 
    for first_level in drug:
        flag = 0
        
        if first_level.tag == '{http://www.drugbank.ca}atc-codes':
            l = level(first_level)
            if not l:
                flag = 1
                break
            result[1] = l[:]
        if first_level.tag == "{http://www.drugbank.ca}half-life":
            h = half_life(first_level)
            if h:
                items += 1
                result[2] = h[:]
            else:
                result[2] =[0.03]

        if first_level.tag == '{http://www.drugbank.ca}experimental-properties':
            e = experimental(first_level)
                   
            if e:
                items += 1
                result[3] =e[:]
            else:
                result[3] =[0.12, 0.3 ]
                
        
        if first_level.tag == '{http://www.drugbank.ca}protein-binding':
            p = protien_binding(first_level)
             
            if p:
                items += 1
                result[4] = p[:]
            else:
                result[4] = [0.3]
        
        if first_level.tag == '{http://www.drugbank.ca}calculated-properties':
            c = calculated_properties(first_level)
            if c:
                items += 1
                result[5] = c
            else:
                result[5] = [0.11, 0.01, 0, 0.99, 0.5]
            
    
    
            
            
    
    if flag==1:
        continue
    elif items>=3:
       count+=1
       try:
           flatten = [item for sublist in result for item in sublist ]
           print (flatten)
           pickle.dump(flatten, output_file, pickle.HIGHEST_PROTOCOL)
       except:
           print(result)
       
print (count)
output_file.close()
    

       