# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 23:29:56 2020

@author: debanjalibiswas

Implementation of a classification model to distinguish between similar Languages

Constants
"""

import os

#We consider 3 similar languages from the DSL dataset (Croatian - hr, Bosnian - bs and Serbian - sr)
lang = ["hr", "bs", "sr"]

#Dataset path (Update correct path)
train_path = "Data/train.txt"
test_path = "Data/test.txt"

#path to store the model checkpoints 
path = "checkpoints" 
checkpoint_path = os.path.join(path,'classifier.pickle')