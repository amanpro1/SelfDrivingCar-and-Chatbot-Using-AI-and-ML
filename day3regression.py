# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:21:38 2019

@author: amanv
"""

import numpy as np
import matplotlib.pyplot as plt

n_pts = 100
x = np.linspace(0,10,n_pts)
y= 5*x+14 +np.random.uniform(-5,5,n_pts)
plt.scatter(x,y,color='b')
plt.show()

from sklearn.linear_model import LinearRegression
#sklearn is package and linear_model is subpackage and LinearRegression is class 

model = LinearRegression()
#below is the conventionfollowed in sklearn package
#if the data contains the single feature reshape to (-1,1)
#if the data contains the single sample(single row) reshape to (1,-1)
model.fit(x.reshape(-1,1),y) 
plt.scatter(x,y,color='b')
plt.plot(x,model.predict(x.reshape(-1,1)),color='r')
plt.show()

un = np.array([9.75])
model.predict(un.reshape(1,-1))

plt.scatter(x,y,color='b')
plt.plot(x,model.predict(x.reshape(-1,1)),color='r')
plt.scatter(un,model.predict(un.reshape(1,-1)),color='g')
plt.show()