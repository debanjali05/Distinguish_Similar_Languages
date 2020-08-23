# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 23:29:56 2020

@author: debanjalibiswas

Implementation of a classification model to distinguish between similar Languages

Training and Testing our classifier model
"""

import os
import pickle
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from data import load_dataset
from sklearn.preprocessing import StandardScaler
from classifier_model import extract_features, train, predict
from utils import train_path, test_path, checkpoint_path, lang

# Loading the train and test sets
print("\t-------Loading Training Set-------")
X_train, y_train = load_dataset(train_path)  #generating the train set
print("Length of Training set:", len(X_train))

print("\t-------Loading Test Set-------")
X_test, y_test = load_dataset(test_path)  #generating the train set
print("Length of Test set:", len(X_test))

# Extracting feature for Train and Test sets using tfidf vectorizer
print("\n\t-------Extracting Features-------")
tfidf = extract_features() 
print("\nFeature Extraction model:", tfidf) 

# generating train set features
X_train_features = tfidf.fit_transform(X_train)

# generating test set features
X_test_features = tfidf.transform(X_test)

# Transforming the extracted features
sc = StandardScaler(with_mean = False)
X_train_features = sc.fit_transform(X_train_features)
X_test_features = sc.fit_transform(X_test_features)

# Training our classifier on training set
print("\n\t-------Start Training------")
classifier = train(X_train_features.toarray(), y_train) 
print("\nEnsemble Classifier Model:", classifier)

# saving classifier
f = open(os.path.join(checkpoint_path,'ensemble_classifier.pickle'), 'wb')
pickle.dump(classifier, f)
f.close()
print("\nModel saved:", os.path.join(checkpoint_path,'ensemble_classifier.pickle')) 
   
print("\t-------End Training-------")

# Testing our classifier on test set
print("\n\t-------Start Testing------")
# loading saved classifier
f = open(os.path.join(checkpoint_path,'ensemble_classifier.pickle'), 'rb')
classifier = pickle.load(f)
f.close()

#calculating accuracy    
print("\n Ensemble Classifier:")
accuracy, confusion_matrix, f1_score = predict(classifier,X_test_features.toarray(),y_test)
print("Accuracy :", accuracy * 100)
print("F1 score :", f1_score)
print("\nConfusion Matrix:")
print(confusion_matrix)

df_cm = pd.DataFrame(confusion_matrix, index = [i for i in lang],columns = [i for i in lang])
plt.figure(figsize=(3,3))
cm = sn.heatmap(df_cm, annot=True)
figure = cm.get_figure()
figure.savefig('confusion_matrix_ensemble.png')
plt.show()

print("\t-------End Testing------")