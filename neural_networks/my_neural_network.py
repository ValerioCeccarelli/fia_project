import numpy as np


class MyNeuralNetworkClassifier:

    def __init__(self, learning_rate: float = 0.01, iterations: int = 100, hidden_neurons: int = 128):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.hidden_neurons = hidden_neurons

        self.theta1 = None
        self.theta2 = None

    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        x_data = np.c_[np.ones((len(x_data), 1)), x_data]
        self.theta1 = np.random.randn(len(x_data[0]), self.hidden_neurons) * 0.1
        self.theta2 = np.random.randn(self.hidden_neurons, 1) * 0.1
        y_data = np.reshape(y_data, (len(y_data), 1))

        m = len(x_data)
        x_data_t = x_data.T

        for i in range(self.iterations):
            z1 = x_data @ self.theta1
            # a1 = 1 / (1 + np.exp(-z1))
            a1 = np.maximum(0, z1)
            z2 = a1 @ self.theta2
            a2 = 1 / (1 + np.exp(-z2)) # relu for regression
            error = a2 - y_data
            gradients2 = 2 / m * a1.T @ error # 2/m * error @ np.where(a2 > 0, 1, 0)
            # gradients1 = 2 / m * x_data_t @ (error @ self.theta2.T * a1 * (1 - a1))
            gradients1 = 2 / m * x_data_t @ (error @ self.theta2.T * np.where(a1 > 0, 1, 0))

            self.theta1 -= self.learning_rate * gradients1
            self.theta2 -= self.learning_rate * gradients2

            if i % 100 == 0:
                print(f"iteration: {i}")

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        x_data = np.c_[np.ones((len(x_data), 1)), x_data]
        results = []
        for x in x_data:
            z1 = x @ self.theta1
            # a1 = 1 / (1 + np.exp(-z1))
            a1 = np.maximum(0, z1)
            z2 = a1 @ self.theta2
            a2 = 1 / (1 + np.exp(-z2))
            results.append(a2)
        return np.array(results)

class MyNeuralNetworkRegressor:

    def __init__(self, learning_rate: float = 0.01, iterations: int = 100, hidden_neurons: int = 128):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.hidden_neurons = hidden_neurons

        self.theta1 = None
        self.theta2 = None

    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        x_data = np.c_[np.ones((len(x_data), 1)), x_data]
        self.theta1 = np.random.randn(len(x_data[0]), self.hidden_neurons) * 0.1
        self.theta2 = np.random.randn(self.hidden_neurons, 1) * 0.1
        y_data = np.reshape(y_data, (len(y_data), 1))

        m = len(x_data)
        x_data_t = x_data.T

        for i in range(self.iterations):
            z1 = x_data @ self.theta1
            # a1 = 1 / (1 + np.exp(-z1))
            a1 = np.maximum(0, z1)
            z2 = a1 @ self.theta2
            a2 = np.maximum(0, z2)
            error = a2 - y_data

            gradients2 = 2/m * np.where(a2.T > 0, 1, 0) @ error
            # gradients1 = 2 / m * x_data_t @ (error @ self.theta2.T * a1 * (1 - a1))
            gradients1 = 2 / m * x_data_t @ (error @ self.theta2.T * np.where(a1 > 0, 1, 0))

            self.theta1 -= self.learning_rate * gradients1
            self.theta2 -= self.learning_rate * gradients2

            if i % 100 == 0:
                print(f"iteration: {i}")

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        x_data = np.c_[np.ones((len(x_data), 1)), x_data]
        results = []
        for x in x_data:
            z1 = x @ self.theta1
            # a1 = 1 / (1 + np.exp(-z1))
            a1 = np.maximum(0, z1)
            z2 = a1 @ self.theta2
            # a2 = 1 / (1 + np.exp(-z2))
            a2 = np.maximum(0, z2)
            results.append(a2)
        return np.array(results)