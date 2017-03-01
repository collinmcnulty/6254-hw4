import sklearn
import numpy as np
from sklearn.datasets import load_boston

boston = load_boston()
X = boston.data
y = boston.target

Xtrain = X[0:399]
Xtest = X[400:]
ytrain = y[0:399]
ytest = y[400:]

# Standardize the data
X_scaler = sklearn.preprocessing.StandardScaler()
Xtrain = X_scaler.fit_transform(Xtrain)
Xtest = X_scaler.transform(Xtest)

# Least Squares

theta = np.linalg.inv(np.dot(Xtrain.transpose(), Xtrain))
theta_element = np.dot(Xtrain.transpose(), ytrain)
theta = np.dot(theta, theta_element)

error = np.linalg.norm(ytest - np.dot(Xtest, theta)) ** 2
error /= len(ytest)  # 447

1
