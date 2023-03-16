import numpy as np

def logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    # Initialize the parameters
    w = np.zeros((X.shape[1], 1))
    b = 0

    # Gradient descent
    for i in range(epochs):
        # Forward propagation
        z = np.dot(X, w) + b
        a = 1 / (1 + np.exp(-z))

        # Backward propagation
        dw = (1 / X.shape[0]) * np.dot(X.T, (a - y))
        db = (1 / X.shape[0]) * np.sum(a - y)

        # Update the parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return w, b

def predict_logistic(X, w, b):
    z = np.dot(X, w) + b
    a = 1 / (1 + np.exp(-z))
    return a

def linear_regression(X, y, learning_rate=0.01, epochs=1000):
    # Initialize the parameters
    w = np.zeros((X.shape[1], 1))
    b = 0

    # Gradient descent
    for i in range(epochs):
        # Forward propagation
        z = np.dot(X, w) + b

        # Backward propagation
        dw = (1 / X.shape[0]) * np.dot(X.T, (z - y))
        db = (1 / X.shape[0]) * np.sum(z - y)

        # Update the parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return w, b

def predict_linear(X, w, b):
    z = np.dot(X, w) + b
    return z

def main():
    # X = np.array([[1, 2], [2, 1], [2, 2], [2, 3], [3, 2], [4, 3], [6, 5], [7, 4], [7, 5], [7, 6], [8, 4], [8, 5], [9, 5]])
    # y = np.array([[0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1]])
    # w, b = logistic_regression(X, y)
    # print(w, b)
    # p = predict(np.array([[1, 2]]), w, b)
    # print(p)
    ins = [x for x in range(1000)]
    X = np.array([[x] for x in ins])
    y = np.array([[20*x+4] for x in ins])
    w, b = linear_regression(X, y)
    print(w, b)
    p = predict_linear(np.array([[3]]), w, b)
    print(p)

if __name__ == '__main__':
    main()