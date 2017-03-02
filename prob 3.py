import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.preprocessing

np.random.seed(2017)
n = 100
xtrain = np.random.rand(n)
ytrain = 0.25 + 0.5*xtrain + np.sqrt(0.1)*np.random.randn(n)
idx = np.random.randint(0,100,10)
ytrain[idx] = ytrain[idx] + np.random.randn(10)
one_col = np.ones([len(xtrain),1])
xtrain = np.c_[xtrain, one_col]
ytrain_original = ytrain

X_scaler = sklearn.preprocessing.StandardScaler()
xtrain = X_scaler.fit_transform(xtrain)

y_scaler = sklearn.preprocessing.StandardScaler()
ytrain = y_scaler.fit_transform(ytrain)


lamb = 1
gamma = lamb * np.identity(2)
#gamma[0,0] = 0
theta = np.linalg.inv(np.dot(xtrain.transpose(), xtrain) + gamma)
theta_element = np.dot(xtrain.transpose(), ytrain)
theta = np.dot(theta, theta_element)

reg = [X_scaler.inverse_transform([theta[0],1])[0], y_scaler.inverse_transform(theta[1])]

prediction2 = np.dot(xtrain,theta[0])[:,0]

error2 = np.linalg.norm(y_scaler.inverse_transform(ytrain) - y_scaler.inverse_transform(prediction2)) ** 2
error2 /= len(ytrain)
# plt.figure()
# plt.plot(xtrain[:,0], ytrain, 'ro')
# plt.plot(xtrain[:,0], prediction2)
# plt.show()

xtrain=X_scaler.inverse_transform(xtrain)[:,0]

from sklearn import linear_model
reg = linear_model.HuberRegressor(epsilon = 2, alpha=0.01)
reg.fit(xtrain.reshape(-1,1),ytrain_original)

pred = reg.predict(xtrain.reshape(-1,1))

# plt.figure()
# plt.plot(xtrain, ytrain_original, 'ro')
# plt.plot(xtrain, pred)
# plt.show()

from sklearn.svm import SVR
reg = SVR(C=10.0, epsilon=0.5, kernel='linear')
reg.fit(xtrain.reshape(-1,1),ytrain_original)

plt.figure()
plt.plot(xtrain, ytrain_original, 'ro')
plt.plot(xtrain, pred)
plt.show()

1