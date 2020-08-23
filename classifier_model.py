# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 23:29:56 2020

@author: debanjalibiswas

Implementation of a classification model to distinguish between similar Languages

Classifier model: soft voting classifier on the ensemble of SVM and Naive Bayes classifiers
using n-gram (2-6) character level Tfidf feature extractor 
"""

from utils import lang
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def extract_features(max_features = 8000):
    """
        To apply tf-idf vectorizer and split data to the test and train data.
        
        max_features: size of feature set
    """
    
    # Tfidf feature extractor on character level n-gram (2-6) 
    tfidf = TfidfVectorizer(ngram_range=(2,6), analyzer= 'char', max_features=max_features)
    
    return tfidf


def train(X_train, y_train):
    """
        Train or classifier model on training set.
        
        X_train: input data
        y_train: data labels
    """
    
    classifier_NB = MultinomialNB() # Naive Bayes Classifier
    classifier_SVM = SGDClassifier(loss = 'log') # SVM classifier using Stochastic Gradient Descent
    
    # soft Voting Classifier on the ensemble of SVM and Naive Bayes
    classifier = VotingClassifier(estimators=[('nb', classifier_NB), ('svm', classifier_SVM)], voting='soft')
    
    # fit the model
    classifier.fit(X_train, y_train)
    
    return classifier


def predict(classifier, X_test, y_test):
    """
        Predict the accuracy of our classifier model on the test set.
        
        classifier: trained classifier
        X_test: test set data
        y_test: test set labels
    """
    
    y_pred = classifier.predict(X_test) # predicting using our model 
    
    accuracy = accuracy_score(y_test, y_pred) # accuracy 
    confusion= confusion_matrix(y_test, y_pred, labels=lang) # confusion matrix
    score_f1 = f1_score(y_test, y_pred, average='weighted') # f1 score
    
    return accuracy, confusion, score_f1