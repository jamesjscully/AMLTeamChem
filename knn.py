#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 17:55:34 2020

@author: nikhil
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Importing the dataset
dataset = pd.read_csv('dataset/dataset.csv', header=None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

labels_main = pd.read_csv('dataset/unique_smells.csv')
labels_main = labels_main.iloc[:, :].values

data_test = pd.read_csv('dataset/test.csv')
smiles_test = data_test['SMILES'].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting classifier to the Training set
classifier = KNeighborsClassifier(n_neighbors = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

dataset_test = pd.read_csv('dataset/dataset_test.csv', header=None)
test_submission_data = dataset_test.iloc[:, :].values

submission = 'SMILES,PREDICTIONS\n'

for smile_test, test_data in zip(smiles_test, test_submission_data):
    distances, indices = classifier.kneighbors(test_data.reshape(1,-1),  n_neighbors=15)
    labels = []
    indices = indices[0]
    for indice in indices:
        print(indice)
        labels.append(labels_main[y[indice]-1][0])
    submission += smile_test + ',"' 
    for s1, s2, s3 in zip(labels[::3],labels[1::3],labels[2::3]): 
        submission += s1 + ',' +  s2 + ',' + s3 + ';'
    submission = submission[:-1] + '"\n'

with open('dataset/submission.csv','w') as file:
    file.write(submission)
        
        
        


