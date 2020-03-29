# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 10:34:52 2019

@author: amanv
"""

#Classification model
#technique applied=logistic regression
import numpy as np
import matplotlib.pyplot as plt
n_pts = 100
x_top = np.random.normal(10,2,n_pts)
y_top = np.random.normal(10,2,n_pts)

x_bot = np.random.normal(5,2,n_pts)
y_bot = np.random.normal(5,2,n_pts)

plt.scatter(x_top,y_top)
plt.scatter(x_bot,y_bot)
plt.show()
top_array = np.array([x_top,y_top]).T
bot_array = np.array([x_bot,y_bot]).T
print(top_array)
print(bot_array[:5,:])
data = np.vstack((top_array,bot_array))
labels = np.matrix(np.append(np.ones(n_pts),np.zeros(n_pts))).T
#print(labels)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(data,labels)
#plotting decision boundary
x_max,x_min=data[:,0].max()+1,data[:,0].min()-1
y_max,y_min=data[:,1].max()+1,data[:,1].min()-1
xx,yy = np.meshgrid(np.linspace(x_min,x_max),(np.linspace(y_min,y_max)))
grid = np.c_[xx.ravel(),yy.ravel()]
pred = model.predict(grid).reshape(xx.shape)
plt.contourf(xx,yy,pred)
plt.scatter(x_top,y_top)
plt.scatter(x_bot,y_bot)
plt.show()

y_pred = model.predict(data)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels,y_pred)
print(cm)
model.score(data,labels)
