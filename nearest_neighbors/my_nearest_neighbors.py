import numpy as np


class MyNearestNeighborsClassifier:

    def __init__(self, distance=1, k_nearest=1):
        assert isinstance(distance, int), "distance should be a integer"
        assert isinstance(k_nearest, int), "k_nearest should be an integer"

        if distance <= 0:
            raise ValueError("distance should be positive and non zero")
        if k_nearest < 1:
            raise ValueError("k_nearest should be positive and non zero")

        self._distance = distance
        self._k_nearest = k_nearest

        self._x_train = None
        self._y_train = None

    def fit(self, x_train: list, y_train: list[bool]):
        assert isinstance(x_train, list) or isinstance(x_train, np.ndarray), "x_train should be a list or a numpy array"
        assert isinstance(y_train, list) or isinstance(y_train, np.ndarray), "y_train should be a list or a numpy array"

        if len(x_train) != len(y_train):
            raise ValueError("x_train and y_train should have the same length")

        if isinstance(x_train, list):
            x_train = np.array(x_train)
        if isinstance(y_train, list):
            y_train = np.array(y_train)

        self._x_train = x_train
        self._y_train = y_train

    def predict(self, x_test: list) -> list[bool]:
        assert isinstance(x_test, list) or isinstance(x_test, np.ndarray), "x_test should be a list or a numpy array"

        if self._x_train is None or self._y_train is None:
            raise ValueError("You should fit the model first")

        if isinstance(x_test, list):
            x_test = np.array(x_test)

        results = []
        for test in x_test:
            temp = self._x_train - test
            dist = np.linalg.norm(temp, ord=self._distance, axis=1)
            min = np.argmin(dist, axis=0)
            results.append(self._y_train[min])
        return results

    def predict_single(self, x_test: list) -> bool:
        assert isinstance(x_test, list) or isinstance(x_test, np.ndarray), "x_test should be a list or a numpy array"

        if self._x_train is None or self._y_train is None:
            raise ValueError("You should fit the model first")

        if isinstance(x_test, list):
            x_test = np.array(x_test)

        best = []
        for train, y in zip(self._x_train, self._y_train):
            distance = np.linalg.norm(x_test - train, ord=self._distance)
            if len(best) < self._k_nearest:
                best.append((distance, y))
            else:
                best.sort(key=lambda x: x[0])
                if distance < best[-1][0]:
                    best[-1] = (distance, y)

        best.sort(key=lambda x: x[0])
        return best[0][1]
