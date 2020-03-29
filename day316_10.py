# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:16:15 2019

@author: amanv
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
data = load_diabetes()
ip= data.data[:,np.newaxis,2]
op=data.target
plt.scatter(ip,op)
plt.show()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(ip,op,test_size=0.2)  

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)
plt.scatter(x_train,y_train,color='b')
plt.scatter(x_test,y_test,color='r')

