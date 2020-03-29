# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 09:40:04 2019

@author: amanv
"""
import numpy as np
import pandas as pd
data = pd.read_csv(r"C:\Users\amanv\Documents\ai&mlclass\Datasets\adult.data",header=None,delimiter=' *, *',engine='python')
data.columns = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','income']
for i in data.columns:
    if data[i].dtype == 'O':
        print(i,':',sum(data[i]=='?'))
        
for i in data.columns:
    if data[i].dtype=='O':
        data[i].replace('?',data.describe(include='all')[i][2],inplace=True)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
data.workclass = le1.fit_transform(data.workclass)
le2 = LabelEncoder()
data.education = le2.fit_transform(data.education)
le3 = LabelEncoder()
data.marital_status = le3.fit_transform(data.marital_status)
le4 = LabelEncoder()
data.occupation = le4.fit_transform(data.occupation)
le5 = LabelEncoder()
data.relationship = le5.fit_transform(data.relationship)
le6 = LabelEncoder()
data.race = le6.fit_transform(data.race)
le7 = LabelEncoder()
data.sex = le7.fit_transform(data.sex)
le8 = LabelEncoder()
data.native_country = le8.fit_transform(data.native_country)
le9 = LabelEncoder()
data.income = le9.fit_transform(data.income)

ip = data.drop(['income'],axis=1)
op = data.income
from sklearn.preprocessing import OneHotEncoder
ohe =OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
ip = ohe.fit_transform(ip).toarray()

from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts = train_test_split(ip,op,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(xtr)
xtr = sc.fit_transform(xtr)
xts = sc.fit_transform(xts)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtr,ytr)

from sklearn.metrics import confusion_matrix

y_pred = model.predict(xts)
print(confusion_matrix(yts,y_pred))
model.score(xts,yts)