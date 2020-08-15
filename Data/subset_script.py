# -*- coding: utf-8 -*-

"""
Created on Fri Aug 14 23:29:56 2020

@author: debanjalibiswas

Generating subset 'train.txt' and 'test.txt' of the DSL (version 4) dataset for 3 similar languages such as 
Croatian, Bosnian, Serbian on both train and test sets

Train set contains 18000 lines for each language so a total of 54000 lines.
Test set contains 1000 lines for each language so a total of 3000 lines.
"""

import codecs

# generating test subset
with codecs.open('dslcc4/DSL-TEST-GOLD.txt',"r","utf-8") as filep:
        lines = filep.readlines()

lines = [line.rstrip() for line in lines]
print("Total Test set length",len(lines)) # 252000 lines

data =[]

for line in lines:
    label = line.split("\t")[1]
    if (label in ['hr','bs','sr']):
        data.append(line)
    
print("Test subset length:",len(data)) # 3000 lines

with open('test.txt', 'w') as f:
    for item in data:
        f.write("%s\n" % item)
        
# generating train subset        
with codecs.open('dslcc4/DSL-TRAIN.txt',"r","utf-8") as filep:
        lines = filep.readlines()

lines = [line.rstrip() for line in lines]
print("Total Train set length",len(lines)) # 54000 lines

data =[]

for line in lines:
    label = line.split("\t")[1]
    if (label in ['hr','bs','sr']):
        data.append(line)
    
print("Train subset length:",len(data))

with open('train.txt', 'w') as f:
    for item in data:
        f.write("%s\n" % item)