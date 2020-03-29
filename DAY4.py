# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 09:23:48 2019

@author: amanv
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n_pts = 100
x = np.linspace(0,10,n_pts)
y= 5*x+14 +np.random.uniform(-5,5,n_pts)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x.reshape(-1,1),y) 
m  = model.coef_
c = model.intercept_
#y = m*x +c
x1 = 0.5
y1 = m*x1 + c
x2 = 10
y2 = m*x2 + c
plt.scatter(x,y,color = 'b')
plt.plot([x1,x2],[y1,y2],color = 'r')
plt.show()
#mse
y_pred = model.predict(x.reshape(-1,1))
mse = ((np.subtract(y_pred,y)**2).sum())/100
print(mse)
r2score=model.score(x.reshape(-1,1),y)
print(r2score)
import seaborn as sns
data = pd.read_excel(r"C:\Users\amanv\Documents\ai&mlclass\Datasets\ENB2012_data.xlsx")
corr = data.corr(method = 'pearson')
plt.figure(figsize=(10,20))
sns.heatmap(corr,cmap='coolwarm',annot=True)
plt.show()
ip = data.drop(['X6','X8','Y1','Y2'],axis=1)
op = data['Y2']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(ip,op,test_size=0.2)  

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)