# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 13:54:55 2019

@author: amanv
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data = pd.read_csv(r"C:\Users\amanv\Documents\ai&mlclass\Datasets\Churn_Modelling.csv")
data2 = data.drop(['RowNumber','CustomerId','Surname'],axis=1)
corr = data2.corr(method = 'pearson')
sns.heatmap(corr,cmap='coolwarm',annot=True)
plt.show()
ip = data2.drop(['Exited'],axis=1)
op = data2['Exited']
ip.head()
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
ip.Gender = le1.fit_transform(ip.Gender)

le2 = LabelEncoder()
ip.Geography = le2.fit_transform(ip.Geography)
ip.tail()
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[1,2])
ip = ohe.fit_transform(ip).toarray()
#from gender to 2 columns and from Geography to 3 columns so 3 extra columns so we are using toarray()
from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts = train_test_split(ip,op,test_size=0.2)
plt.boxplot(xtr)
plt.show()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(xtr)
xtr = sc.transform(xtr)
xts = sc.transform(xts)
plt.boxplot(xtr)
plt.show()

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtr,ytr)

from sklearn.metrics import confusion_matrix
y_pred = model.predict(xts)
print(confusion_matrix(yts,y_pred))
model.score(xts,yts)