from numpy import array
from numpy.linalg import pinv
from matplotlib import pyplot
from sklearn import datasets
import numpy as np

data = datasets.load_diabetes()

X = np.append(np.array(data.data), np.ones((442,1)), axis=1)
y = np.array(data.target)

# form A and B
n,m = X.shape
A = np.zeros((m, m))
B = np.zeros((m, 1))
for i in range(n):
	A = A + np.matmul(X[i].reshape(m,1),np.transpose(X[i].reshape(m,1)))
	B = B + y[i] * X[i].reshape(m,1)

# calculate coefficients
coeff = np.matmul(pinv(A),B)
print "coefficients - "
print(coeff)

# predict using coefficients
pred = X.dot(coeff)

#calculate loss
loss = 0
for i in range(n):
	loss = (pred[i] - y[i])*(pred[i] - y[i])
loss = loss/n

print "loss - "  ,
print loss

# plot data and predictions
# pyplot.scatter(X, y)
# pyplot.plot(X, yhat, color='red')
# pyplot.show()
