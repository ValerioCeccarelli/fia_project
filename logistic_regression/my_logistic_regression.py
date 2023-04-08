import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class MyLogisticClassifier:

    def __init__(self, learning_rate: float = 0.01, iterations: int = 100, l1_penalty: float = 0.0,
                 l2_penalty: float = 0.0):
        assert isinstance(learning_rate, float) or isinstance(learning_rate, int), "learning_rate must be a number"
        assert isinstance(iterations, int), "iterations must be an integer"
        assert isinstance(l1_penalty, float) or isinstance(l1_penalty, int), "l1_penalty must be a number"
        assert isinstance(l2_penalty, float) or isinstance(l2_penalty, int), "l2_penalty must be a number"

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.theta = None

    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        assert isinstance(x_data, np.ndarray), "x_data must be a numpy array"
        assert isinstance(y_data, np.ndarray), "y_data must be a numpy array"

        x_data = np.c_[np.ones((len(x_data), 1)), x_data]
        self.theta = np.zeros((len(x_data[0]), 1))
        y_data = np.reshape(y_data, (len(y_data), 1))

        m = len(x_data)
        x_data_t = x_data.T

        for _ in range(self.iterations):
            #TODO: check if this is correct
            y_pred = sigmoid(x_data @ self.theta)
            error = y_pred - y_data
            gradients = 1 / m * x_data_t @ error

            l1_penalty = 0
            if self.l1_penalty != 0:
                l1_penalty = self.l1_penalty * np.sign(self.theta)
            l2_penalty = 0
            if self.l2_penalty != 0:
                l2_penalty = self.l2_penalty * self.theta

            self.theta -= self.learning_rate * (gradients + l1_penalty + l2_penalty)

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        assert isinstance(x_data, np.ndarray), "x_data must be a numpy array"

        x_data = np.c_[np.ones((len(x_data), 1)), x_data]
        return np.round(sigmoid(x_data @ self.theta))

    def predict_single(self, x_data: np.ndarray) -> bool:
        assert isinstance(x_data, np.ndarray), "x_data must be a numpy array"

        x_data = np.c_[np.ones((len(x_data), 1)), x_data]
        return np.round(sigmoid(x_data @ self.theta)) >= 0.5
