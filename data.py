# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 23:29:56 2020

@author: debanjalibiswas

Implementation of a classification model to distinguish between similar Languages

Loading and Preprocessing Dataset
"""

import re
import string
import codecs

def preprocessing(line):
    """
        data preprocessing step: 
            1) Convert to lowercase
            2) Remove line number, digits, punctuations and extra spaces
    
        line: each sentence from the dataset (string)
    """
    
    translate_table = dict((ord(char), None) for char in string.punctuation)
    
    line = line.rstrip()
    line = line.lower() #converting the text to lowercase 
    line = re.sub(r"\d+", "", line) #removing any digits present in the text
    line = line.translate(translate_table)  #removing all punctuations
    line = re.sub(' +',' ',line) #removing extra spaces
     
    return line #preprocessed text

def load_dataset(path):
    """
        Building the dataset for individual language
    
        path: path to the text
        language: language label (string)
    """
    
    X = y = []
    
    #Reading the data text files in unicode
    with codecs.open(path,"r","utf-8") as filep:
        lines = filep.readlines()
        
    lines = [preprocessing(line) for line in lines ]
    X = [line.split("\t")[0] for line in lines]
    y = [line.split("\t")[1] for line in lines]
          
    return X, y 
