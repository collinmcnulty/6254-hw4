from __future__ import division
import numpy as np
import math
np.random.seed(2017)
n = 100
xtrain = np.random.rand(n)
ytrain = np.sin(9*xtrain) + np.sqrt(1/3.0)*np.random.randn(n)

xtest = np.linspace(0,1,1001)
ytest = np.sin(9*xtest)

def errorcalc(yhat, y):
    error2 = np.linalg.norm(yhat - y) ** 2
    error2 /= len(y)
    return error2

K=np.empty([100,100])
gamma = 10;
lamb=1

def kernel(gamma, x1, x2):
    return math.exp(-gamma * np.linalg.norm(x1 - x2))

for i, x in enumerate(xtrain):
    for j, xl in enumerate(xtrain):
        K[i,j]=kernel(gamma, x, xl)

inv = np.linalg.inv(K+lamb*np.identity(len(xtrain)))
fhat_preface = np.dot(ytrain.transpose(),inv)


def kernel_predict(fhatp, xnew, xtrain):
    f = []
    for element in xnew:
        k = []
        for x in xtrain:
            k.append(kernel(gamma, x, element))
        f.append(np.dot(fhatp,k))
    return f

prediction = kernel_predict(fhat_preface, xtest, xtrain)
error1 = errorcalc(prediction, ytest) #0.05

from sklearn.svm import SVR
reg = SVR(C=17, epsilon=0.3, kernel='rbf', gamma=5)
reg.fit(xtrain.reshape(-1,1),ytrain)

ypred=reg.predict(xtest.reshape(-1,1))

error_svr = errorcalc(ypred, ytest) #0.015
1

