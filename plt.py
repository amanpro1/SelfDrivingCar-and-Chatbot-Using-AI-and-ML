import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()
ip = iris.data[:,2:4]
op = iris.target
plt.scatter(ip[:,0],ip[:,1],c=op)
plt.show()
x_max,x_min = ip[:,0].max(),ip[:,0].min()
y_max,y_min = ip[:,1].max(),ip[:,1].min()
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(ip,op)
xx,yy = np.meshgrid(np.linspace(x_min,x_max),(np.linspace(y_min,y_max)))
grid = np.c_[xx.ravel(),yy.ravel()]
pred = model.predict(grid).reshape(xx.shape)
plt.contourf(xx,yy,pred)
plt.scatter(ip[:,0],ip[:,1],c=op)
plt.show()
from sklearn.naive_bayes import GaussianNB

modelNB = GaussianNB()
modelNB.fit(ip,op)
predNB = modelNB.predict(grid).reshape(xx.shape)
plt.contourf(xx,yy,predNB)
plt.scatter(ip[:,0],ip[:,1],c=op)
plt.title('Naive Bayes')
plt.show()