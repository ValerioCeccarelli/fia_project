import random


# def f(w, x):
#     res = 0
#     for i in range(len(x)):
#         res += w[i] * x[i]
#     return res
#
#
# ins_x = [(x, 1) for x in range(100)]
# random.shuffle(ins_x)
#
# ins_y = [f((30,70), x) for x in ins_x]
#
# w = [0, 0]
# L = 0.0001  # The learning Rate
# epochs = 10000  # The number of iterations to perform gradient descent
#
# n = float(len(ins_x))  # Number of elements in X
#
# # Performing Gradient Descent
# for i in range(epochs):
#     for j in range(len(ins_x)):
#         x = ins_x[j]
#         y = ins_y[j]
#         Y_pred = f(w, x)
#         for k in range(len(w)):
#             w[k] -= L * (Y_pred - y) * x[k]
#
# print(w)

# w = [0, 0]
#
# for i in range(epochs):
#     for j in range(len(ins_x)):
#         x = ins_x[j]
#         y = ins_y[j]
#         Y_pred = f(w, x)
#         # for k in range(len(w)):
#         #     w[k] -= L * (Y_pred - y) * x[k]
#
#         w -= L * (Y_pred - y) * x

def f(w, x):
    res = 0
    for i in range(len(x)):
        res += w[i] * x[i]
    return res


ins_x = [(x, 1) for x in range(100)]
random.shuffle(ins_x)

w_vero = (5, 8)
ins_y = [f(w_vero, x) for x in ins_x]

w = [0, 0]
L = 0.01  # The learning Rate
epochs = 100  # The number of iterations to perform gradient descent

n = float(len(ins_x))  # Number of elements in X

# Performing Gradient Descent
for epochs in range(epochs):
    for i in range(len(w)):
        sum = 0
        for j in range(len(ins_x)):
            X_j = ins_x[j]
            Y_j = ins_y[j]
            sum += (Y_j - f(w, X_j)) * X_j[i]
        w[i] += L * sum

print(w)