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

    def predict(self, x_tests: list) -> list[bool]:
        assert isinstance(x_tests, list) or isinstance(x_tests, np.ndarray), "x_test should be a list or a numpy array"

        if self._x_train is None or self._y_train is None:
            raise ValueError("You should fit the model first")

        if isinstance(x_tests, list):
            x_tests = np.array(x_tests)

        results = []
        for test in x_tests:
            result = self.predict_single(test)
            results.append(result)
        return results

    def predict_single(self, x_test: list) -> bool:
        assert isinstance(x_test, list) or isinstance(x_test, np.ndarray), "x_test should be a list or a numpy array"

        if self._x_train is None or self._y_train is None:
            raise ValueError("You should fit the model first")

        if isinstance(x_test, list):
            x_test = np.array(x_test)

        temp = self._x_train - x_test
        dist = np.linalg.norm(temp, ord=self._distance, axis=1)
        min = np.argpartition(dist, self._k_nearest)[:self._k_nearest]
        n_positive = len([x for x in self._y_train[min] if x])
        result = True if n_positive > self._k_nearest / 2 else False
        return result
