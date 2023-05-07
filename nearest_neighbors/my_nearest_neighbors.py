import numpy as np
from scipy.spatial import distance

class MyNearestNeighborsClassifier:
    """
    This class implements a K Nearest Neighbors classifier\n
    A prediction is made by calculating the distance between the point to predict and the
    training points and then taking the majority vote of the k nearest points (k is 5 by default).\n
    The distance is a Minkowski distance with p=2 by default (Euclidean distance).\n
    """

    def __init__(self, distance=2, k_nearest=5):
        """
        :param distance: represents the p parameter of the Minkowski distance
        :param k_nearest: the number of nearest neighbors to take into account
        """
        assert isinstance(distance, int), "distance should be a integer"
        assert isinstance(k_nearest, int), "k_nearest should be an integer"

        if distance < 1:
            raise ValueError("distance should be positive and non zero")
        if k_nearest < 1:
            raise ValueError("k_nearest should be positive and non zero")

        self._distance = distance
        self._k_nearest = k_nearest

        self._x_train = None
        self._y_train = None

    def fit(self, x_train: list, y_train: list[bool]):
        """
        This method doesn't contain any particular logic, it only stores the training data
        """
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
        """
        This method predicts the class of each element in x_tests
        """
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
        """
        This method predicts the class of a single element
        """
        assert isinstance(x_test, list) or isinstance(x_test, np.ndarray), "x_test should be a list or a numpy array"

        if self._x_train is None or self._y_train is None:
            raise ValueError("You should fit the model first")

        if isinstance(x_test, list):
            x_test = np.array(x_test)

        # differences = self._x_train - x_test
        # distances = np.linalg.norm(differences, ord=self._distance, axis=1)
        distances = distance.cdist([x_test], self._x_train, metric='minkowski', p=self._distance)[0]
        min = np.argpartition(distances, self._k_nearest)[:self._k_nearest]
        n_positive = len([x for x in self._y_train[min] if x])
        result = True if n_positive > self._k_nearest / 2 else False
        return result


class MyNearestNeighborsRegressor:
    """
    This class implements a K Nearest Neighbors regressor\n
    A prediction is made by calculating the distance between the point to predict and the
    training points and then taking the mean value of the k nearest points (k is 5 by default).\n
    The distance is a Minkowski distance with p=2 by default (Euclidean distance).\n
    """

    def __init__(self, distance=2, k_nearest=5):
        """
        :param distance: represents the p parameter of the Minkowski distance
        :param k_nearest: the number of nearest neighbors to take into account
        """
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

    def fit(self, x_train: list, y_train: list):
        """
        This method doesn't contain any particular logic, it only stores the training data
        """
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

    def predict(self, x_tests: list) -> list[float]:
        """
        This method predicts the class of each element in x_tests
        """
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

    def predict_single(self, x_test: list) -> float:
        """
        This method predicts the class of a single element
        """
        assert isinstance(x_test, list) or isinstance(x_test, np.ndarray), "x_test should be a list or a numpy array"

        if self._x_train is None or self._y_train is None:
            raise ValueError("You should fit the model first")

        if isinstance(x_test, list):
            x_test = np.array(x_test)

        # differences = self._x_train - x_test
        # distances = np.linalg.norm(differences, ord=self._distance, axis=1)
        distances = distance.cdist([x_test], self._x_train, metric='minkowski', p=self._distance)[0]
        k_min = np.argpartition(distances, self._k_nearest)[:self._k_nearest]
        mean = np.mean(self._y_train[k_min])
        return mean
