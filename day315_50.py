# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:44:10 2019

@author: amanv
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r"C:\Users\amanv\Documents\ai&mlclass\Datasets\auto-mpg.csv",header=None)
data.columns=['mpg','cylinder','displacement','horsepower','weight','acceleration','model_year','origin','car_name']
for i in ['horsepower']:
    data[i].replace('?',data.describe(include='all')[i][2],inplace=True)
for i in ['horsepower']:
    print(sum(data[i]=='?'))
data['horsepower']= data.horsepower.astype(float)
ip = data.drop(['mpg','car_name'],axis=1)
#ip.head(2)
op=data['mpg']
#train_test_split is a method not class(class name start with capital letter that  a convention)
from sklearn.model_selection import train_test_split
#order in writingx_train,x_test,y_train,y_test should be maintained as this is array
x_train,x_test,y_train,y_test = train_test_split(ip,op,test_size=0.2)  #splitting randomly

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

model.score(x_test,y_test)  #matching with actual value on 79% times
