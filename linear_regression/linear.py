import random

import numpy as np


# def f(x):
#     return 60 * x + -300
#
#
# ins = [x for x in range(100)]
# random.shuffle(ins)
#
# X = np.array(ins)
# Y = np.array([f(x) for x in ins])
#
#
# # Building the model
# m = 0
# c = 0
#
# L = 0.0001  # The learning Rate
# epochs = 100000  # The number of iterations to perform gradient descent
#
# n = float(len(X))  # Number of elements in X
#
# # Performing Gradient Descent
# for i in range(epochs):
#     Y_pred = m * X + c  # The current predicted value of Y
#     D_m = (-2 / n) * sum(X * (Y - Y_pred))  # Derivative wrt m
#     D_c = (-2 / n) * sum(Y - Y_pred)  # Derivative wrt c
#     m = m - L * D_m  # Update m
#     c = c - L * D_c  # Update c
#
# print(m, c)

# def f(w, x):
#     sum = 0
#     for i in range(len(x)):
#         sum += w[i] * x[i]
#     return sum
#
#
# ins = [(x, 1) for x in range(100)]
# random.shuffle(ins)
#
# w_vera = (60, -300)
# X = np.array(ins)
# Y = np.array([f(w_vera, x) for x in ins])
#
#
# # Building the model
# w = np.array([0, 0])
#
# L = 0.0001  # The learning Rate
# epochs = 100000  # The number of iterations to perform gradient descent
#
# n = float(len(X))  # Number of elements in X
#
# # Performing Gradient Descent
# for i in range(epochs):
#     Y_pred = w * X  # The current predicted value of Y
#     D_m = (-2 / n) * sum(X * (Y - Y_pred))  # Derivative wrt m
#     D_c = (-2 / n) * sum(Y - Y_pred)  # Derivative wrt c
#     m = m - L * D_m  # Update m
#     c = c - L * D_c  # Update c
#
# print(m, c)

# X = np.array([[2., 1., 1.],
#               [4., 2., 1.],
#               [6., 3., 1.]])
# y = np.array([[1.],[2.],[3.]])
# w = np.array([[0.], [0.], [0.]])
# L = 0.005
#
# for epoch in range(100):
#     mom = np.matmul(X, w)
#     mom = mom - y
#     d_j = 1 / len(X) * np.matmul(X.T, mom)
#     w -= L * d_j
#
# print(w)
#
# print(np.matmul(np.array([[8., 4., 1.]]), w))

import numpy as np
from common.read_dataset import read_dataset_with_pandas

df = read_dataset_with_pandas("../dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv")

data = np.array(df)
x_data = list(data[:, :-1])
for i in range(len(x_data)):
    x_data[i] = list(x_data[i])
    x_data[i].append(1)

x_data = np.array(x_data)

Y_data = np.array([data[:, -1]]).T
print(Y_data.shape)



w = np.array([[0.0] for _ in range(len(x_data[0]))])

L = 0.005
for epoch in range(1):
    for i in range(0, len(x_data), 100):
        cache = x_data[i: i+100]
        mom = np.matmul(cache, w)
        mom = mom - Y_data[i: i+100]
        d_j = 1 / 100 * np.matmul(cache.T, mom)
        w -= L * d_j

print(w)