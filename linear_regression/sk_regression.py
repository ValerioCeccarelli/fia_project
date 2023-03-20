from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import numpy as np

from common.read_dataset import read_dataset_with_pandas


def read_dataset_with_numpy(file_path: str) -> (np.ndarray, np.ndarray):
    df = read_dataset_with_pandas(file_path)
    data = np.array(df)
    x_data = data[:, :-1]
    y_data = data[:, -1]
    return x_data, y_data

x_data, y_data = read_dataset_with_numpy("../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv")
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)


# reg = LinearRegression()
# reg: LinearRegression = reg.fit(x_train, y_train)
#
# print(reg.coef_)
# print(reg.intercept_)
#
# start = time.time()
# p = reg.predict(x_test)
# end = time.time()
# val = mean_squared_error(y_test, p)
# print(f"mean squared error: {val} in {end - start} seconds")

def generateXvector(X):
    """ Taking the original independent variables matrix and add a row of 1 which corresponds to x_0
        Parameters:
          X:  independent variables matrix
        Return value: the matrix that contains all the values in the dataset, not include the outcomes variables.
    """
    vectorX = np.c_[np.ones((len(X), 1)), X]
    return vectorX

def theta_init(X):
    theta = np.random.randn(len(X[0])+1, 1)
    return theta

def Multivariable_Linear_Regression(X,y,learningrate, iterations):
    y_new = np.reshape(y, (len(y), 1))
    vectorX = generateXvector(X)
    theta = theta_init(X)
    m = len(X)
    for i in range(iterations):
        y_stimated = vectorX.dot(theta)
        y_error = y_stimated - y_new
        temp = vectorX.T.dot(y_error)
        gradients = 2/m * temp
        theta = theta - learningrate * gradients
    return theta

start = time.time()
theta = Multivariable_Linear_Regression(x_train, y_train, 0.00001, 1000)
end = time.time()

print(theta)

# ins_x = np.array([[i] for i in range(100)])
# ins_y = np.array([20*x+30 for x in ins_x])
#
# start = time.time()
# theta = Multivariable_Linear_Regression(ins_x, ins_y, 0.0001, 100000)
# end = time.time()

print(f"theta: {theta} in {end - start} seconds")