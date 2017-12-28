#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 18:33:19 2017

@author: ranjeetapegu
"""

import numpy as np
import pandas as pd

# creating simulated dataset
x1 = np.random.uniform(0, 100, 100)
x2 = np.random.uniform(0,100,100)
def bound(x):
    X= 50+(x-2*(x-20)+3*(x-50)**3/8000)
    return X
y = np.where(x2 > bound(x1),0,1)

# creating the Data Frame
df1 = pd.DataFrame(x1,columns=['x1'])
df2 = pd.DataFrame(x2,columns=['x2'])
df3 = pd.DataFrame(y,columns =['y'])
df = pd.concat([df1, df2,df3], axis=1)
print(df.head())

y=df.iloc[:,2]
X =df.iloc[:,0:2]
print(X.head())


# LINEAR SUPPORT VECTOR MACHINE

from sklearn.svm import LinearSVC
svc= LinearSVC()
svc.fit(X,y)

print(svc.coef_)

print(svc.intercept_)

y_pred =svc.predict(X)
y_pred

## ploting the svm in 2d
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
#%matplotlib inline

plot_decision_regions(X.values, y.values, clf=svc,
                      res=0.02, legend=2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('SVM on Simulated Dataset')
plt.show()

## from sklearn.metrics import accuracy_score and confusion matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
print(accuracy_score(y, y_pred))
confusion_matrix(y,y_pred)


