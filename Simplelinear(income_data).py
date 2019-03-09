# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 16:48:45 2018

@author: hp
"""

#SIMPLE LINEAR REGRESSION

#IMPORTING THE LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt

#IMPORTING THE DATASET
dataset=pd.read_csv("Income_Data.csv") 
features=dataset.iloc[:,:-1].values
labels=dataset.iloc[:,1].values

#splitting dataset
from sklearn.cross_validation import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)

#REGRESSION STEPS
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(features_train,labels_train)

#PREDICTING THE TEST VALUES USING OUR MODEL
labels_pred=regressor.predict(features_test)

#SCORE OF MODEL
Score=regressor.score(features_test,labels_test)

#PLOTTING GRAPHS
plt.scatter(features_train,labels_train,color='red')
plt.plot(features_train,regressor.predict(features_train),color='blue')
plt.title("experience vs salary")
plt.xlabel("EXPERIENCE")
plt.ylabel("SALARY")
plt.show()

plt.scatter(features_test,labels_test,color='red')
plt.plot(features_train,regressor.predict(features_train),color='blue')
plt.title("experience vs salary")
plt.xlabel("EXPERIENCE")
plt.ylabel("SALARY")
plt.show()
