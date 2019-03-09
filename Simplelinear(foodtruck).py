# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 12:09:34 2018

@author: hp
"""

#SIMPLE LINEAR REGRESSION

#IMPORT THE LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv("Foodtruck.csv")
features=dataset.iloc[:,:-1].values
labels=dataset.iloc[:,1].values

#splitting the dataset
from sklearn.cross_validation import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)

#performing regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(features_train,labels_train)

#predicting 
lables_pred=regressor.predict(features_test)
regressor.predict(3.073)
#score
score=regressor.score(features_test,labels_test)
