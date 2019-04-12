# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:25:33 2019

@author: Nidhi
"""

#Data Visualization
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
'''
dataset.columns = [c.replace(' ', '_') for c in dataset.columns]
data_test.columns = [c.replace(' ', '_') for c in data_test.columns]
print(data_test.columns)

sns.set(style="darkgrid")
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
Y_distribution = sns.countplot(ax=ax,x="Aggregate_rating", data=dataset)
plt.title("Y_distribution")
Y_distribution.figure.savefig("Y_distribution.png")

city = dataset.City.unique()
for index, i in enumerate(city):
    dataset.City = dataset.City.replace({i: index+1})
    print(i, index+1)
#print(dataset.City.unique())
#print(data_test.City.unique())
output = open('cuisines_index.txt','w')
cuisines = dataset.Cuisines.unique()
for index, i in enumerate(cuisines):
    dataset.Cuisines = dataset.Cuisines.replace({i: index+1})
    output.write("%s " % i)
    output.write("%s\n" % str(index+1))
#print(dataset.Cuisines.unique())
print(data_test.Cuisines.unique())

currency = dataset.Currency.unique()
for index, i in enumerate(currency):
    dataset.Currency = dataset.Currency.replace({i: index+1})
    print(i, index)
print(dataset.Currency.unique())
print(data_test.Currency.unique())

dataset.to_csv('new_train.csv')'''

X_train = dataset.drop('Aggregate_rating', axis=1)
y_train = dataset.Aggregate_rating
X_pred = data_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Create a random dataset
# Fit regression model
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score
dtree = DecisionTreeRegressor(criterion = "mse", random_state = 1, max_depth=5, min_samples_leaf=10)
dtree.fit(X_train, y_train)
dtree_pred = dtree.predict(X_test)

print("Mean squared error: %.2f"
      % mean_squared_error(y_test, dtree_pred))
y_pred = dtree.predict(X_pred)
for i in range(len(y_pred)):
    print(y_pred[i])'''
