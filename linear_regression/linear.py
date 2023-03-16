import numpy as np

X = np.array([1, 2, 2, 2, 3, 4, 6, 7, 7, 7, 8, 8, 9])
Y = np.array([1, 2, 2, 2, 3, 4, 6, 7, 7, 7, 8, 8, 9])


# Building the model
m = 0
c = 0

L = 0.0001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(X))  # Number of elements in X

# Performing Gradient Descent
for i in range(epochs):
    Y_pred = m * X + c  # The current predicted value of Y
    D_m = (-2 / n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2 / n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c

print(m, c)