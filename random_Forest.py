# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 01:37:52 2019

@author: Nidhi
"""


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with codecs.open('new_train.csv', 'r', encoding='utf-8',
                 errors='ignore') as fdata:
    dataset = pd.read_csv(fdata)
    
with codecs.open('new_test.csv', 'r', encoding='utf-8',
                 errors='ignore') as fdata:
    data_test = pd.read_csv(fdata)

X_train = dataset.drop('Aggregate_rating', axis=1)
y_train = dataset.Aggregate_rating
X_pred = data_test
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)
# Create a random dataset
# Fit regression model
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
dtree = RandomForestRegressor(n_estimators=500, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
dtree.fit(X_train, y_train)

#y_pred = dtree.predict(X_test)
#for i in range(len(y_pred)):
#    print(y_pred[i])
#print("Mean squared error: %.2f"
#S      % mean_squared_error(y_test, y_pred))
y_pred = dtree.predict(X_test)
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
y_pred = dtree.predict(X_pred)
for i in range(len(y_pred)):
    print(y_pred[i])

