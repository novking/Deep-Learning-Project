# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 17:40:54 2016

@author: yiz613
"""

import re
import json
level_dict = {}
starting_number = 0


start = starting_number
with open("ATC.txt") as f:
    for line in f.readlines():
        drug_description = re.findall(r'^(.+) \(', line)
        try:
            string = drug_description[0]
            level_dict[string]=start
            start += 1
        except:
            pass
        
with open("output.txt", 'w') as outfile:
    json.dump(level_dict, outfile)