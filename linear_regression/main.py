from common.read_dataset import read_dataset_for_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

x_data, y_data = read_dataset_for_classification("../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv")

x_data = StandardScaler().fit(x_data).transform(x_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

sk_regressor = SGDRegressor(penalty=None, max_iter=100000000, shuffle=False, learning_rate="constant", eta0=0.00001,
                            early_stopping=False)
sk_regressor = sk_regressor.fit(x_train, y_train)
sk_y_pred = sk_regressor.predict(x_test)

print(f"scikit RMSE: {metrics.mean_squared_error(y_test, sk_y_pred, squared=False)}")


def generateXvector(X):
    vectorX = np.c_[np.ones((len(X), 1)), X]
    return vectorX


def theta_init(X):
    theta = np.random.randn(len(X[0]) + 1, 1)
    return theta


def Multivariable_Linear_Regression(X, y, learningrate, iterations):
    y_new = np.reshape(y, (len(y), 1))
    vectorX = generateXvector(X)
    theta = theta_init(X)
    m = len(X)
    for i in range(iterations):
        gradients = 2 / m * vectorX.T.dot(vectorX.dot(theta) - y_new)
        theta = theta - learningrate * gradients
    return theta


theta = Multivariable_Linear_Regression(x_train, y_train, 0.01, 10000)


def predict(X, theta):
    vectorX = generateXvector(X)
    return vectorX.dot(theta)


y_pred = predict(x_test, theta)

print(f"RMSE: {metrics.mean_squared_error(y_test, y_pred, squared=False)}")

exit(0)

