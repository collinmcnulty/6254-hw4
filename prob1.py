import sklearn
import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

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

y_scaler = sklearn.preprocessing.StandardScaler()
ytrain = y_scaler.fit_transform(ytrain)
ytest = y_scaler.transform(ytest)

# Least Squares

theta = np.linalg.inv(np.dot(Xtrain.transpose(), Xtrain))
theta_element = np.dot(Xtrain.transpose(), ytrain)
theta = np.dot(theta, theta_element)

prediction1 = np.dot(Xtest, theta)
# plt.figure()
# plt.plot(prediction1, ytest,'ro')
# plt.show()
error = np.linalg.norm(y_scaler.inverse_transform(ytest) - y_scaler.inverse_transform(prediction1)) ** 2
error /= len(ytest)  # 41

# Ridge Regression
# Xholdout = Xtrain[0:49]
# Xtrainridge = Xtrain[50:399]


lamb = 50
gamma = lamb * np.identity(13)
gamma[0,0] = 0
theta = np.linalg.inv(np.dot(Xtrain.transpose(), Xtrain) + gamma)
theta_element = np.dot(Xtrain.transpose(), ytrain)
theta = np.dot(theta, theta_element)

prediction2 = np.dot(Xtest,theta)

error2 = np.linalg.norm(y_scaler.inverse_transform(ytest) - y_scaler.inverse_transform(prediction2)) ** 2
error2 /= len(ytest)

# Lasso
from sklearn import linear_model
reg = linear_model.Lasso(alpha=4)
reg.fit(Xtrain,ytrain)
prediction3 = reg.predict(Xtest)

error3 = np.linalg.norm(y_scaler.inverse_transform(ytest) - y_scaler.inverse_transform(prediction2)) ** 2
error3 /= len(ytest)
1
