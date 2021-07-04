# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:56:04 2020

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

dataset=pd.read_csv('jobclassinfo2.csv')
print(dataset)

dataset.drop(["JobFamily","JobFamilyDescription","JobClass","JobClassDescription","PayGrade"],axis=1,inplace=True)
X= dataset.iloc[: , [0,1,3,4,5,6,7]].values
y=dataset.iloc[:,2].values

X = preprocessing.scale(X)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#logistic regreesion

logi=LogisticRegression()
logi.fit(X_train,y_train)
predictions=logi.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions)) 



#KNeighbor classifier
classifier_knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier_knn.fit(X_train,y_train)
y_pred=classifier_knn.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))



#decision tree
from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
cart.fit(X_train, y_train)
predictions = cart.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions)) 



#svm
sv = SVC(kernel='linear')
sv.fit(X_train, y_train)
predictions = sv.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))



#random forest
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
