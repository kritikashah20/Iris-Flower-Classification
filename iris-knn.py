# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:54:21 2020

@author: Kritika Shah
"""


# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Importing the dataset
dataset = pd.read_csv('iris.data',sep= ',' )
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting K-NN to the Training set
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Accuracy
acc = classifier.score(X_test, y_test)
print("\nAccuracy = ", acc*100,"%")

'''# Predicting the Test set results
knnPickle = open('knnpickle_file', 'wb')

pickle.dump(classifier, knnPickle)

# Loading the model
loaded_model = pickle.load(open('knnpickle_file', 'rb'))
result = loaded_model.predict(X_test)
'''
'''# Displaying the predicted and actual values 
print("\n0 = star , 1 = galaxy")
print("\nData: [DistinctValues  StdDevGradient_X] \n")
for x in range(len(result)):
    print("Predicted: ", result[x], " Data: ", X_test[x], " Actual: ", y_test[x])'''

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, classifier.predict(X_test))
print("\n The Confusion Matrix:")
print(cm)
