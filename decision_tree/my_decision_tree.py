import numpy as np
import math


class TreeNode:
    """
    A node in a decision tree, this class is used only to represent the tree structure (it does not contain any logic)
    """
    def __init__(self, feature: int, value: float, left, right):
        assert isinstance(left, bool) or isinstance(left, TreeNode), f"left must be a TreeNode object or a boolean value but is: {type(left)}"
        assert isinstance(right, bool) or isinstance(right, TreeNode), "right must be a TreeNode object or a boolean value"
        assert isinstance(feature, int), "feature must be a int"
        assert isinstance(value, float), "value must be a float"

        self.feature = feature
        self.value = value
        self.left = left
        self.right = right


def binary_entropy(p: float) -> float:
    """
    Return the entropy of a binary source with 2 symbols of probability "p" and "1-p"

    :param p: should be a value between 0 and 1
    """

    if p < 0 or p > 1:
        raise ValueError(f"p must be a value between 0 and 1, currently is: {p}")

    # confirmed by plot this function
    if p == 0 or p == 1:
        return 0

    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def h_entropy(p: float, n: float) -> float:
    """
    Return the binary entropy of p / (p + n)

    :param p: number of positive sample
    :param n: number of negative sample
    """
    return binary_entropy(p / (p + n))


def plurality_value(y_train: list[bool]) -> bool:
    """
    return the most frequently repeated boolean element

    :param y_train: a list containing boolean value
    """
    assert isinstance(y_train, list), "boolean_list should be a list object"

    if len(y_train) == 0:
        raise ValueError("y_train should not be empty")

    positive = len([y for y in y_train if y])
    if positive >= len(y_train) / 2:
        return True

    # TODO: the pseudocode says to choose random if there is a tie
    return False


def find_median(x_train: list, feature: int):
    assert isinstance(x_train, list) or isinstance(x_train, np.ndarray), "x_train should be a list or a numpy array"
    assert isinstance(feature, int), "feature should be an int"

    x_train = sorted(x_train, key=lambda x: x[feature])
    if len(x_train) % 2 != 0:
        return x_train[len(x_train) // 2][feature]

    right = x_train[len(x_train) // 2][feature]
    left = x_train[len(x_train) // 2 - 1][feature]
    return (left + right) / 2


def make_partitions(x_train: list, y_train: list[bool], median: float, feature: int):
    assert isinstance(x_train, list) or isinstance(x_train, np.ndarray), "x_train should be a list or a numpy array"
    assert isinstance(y_train, list) or isinstance(y_train, np.ndarray), "y_train should be a list or a numpy array"
    assert isinstance(median, float), "median should be a float"
    assert isinstance(feature, int), "feature should be an int"

    x_train_left = []
    y_train_left = []
    x_train_right = []
    y_train_right = []
    for x, y in zip(x_train, y_train):
        if x[feature] <= median:
            x_train_left.append(x)
            y_train_left.append(y)
        else:
            x_train_right.append(x)
            y_train_right.append(y)
    return x_train_left, y_train_left, x_train_right, y_train_right


def remainder(x_train: list, y_train: list[bool], feature: int) -> float:
    """
    Return the remainder of the partition using the feature
    """
    assert isinstance(x_train, list) or isinstance(x_train, np.ndarray), "x_train should be a list or a numpy array"
    assert isinstance(y_train, list) or isinstance(y_train, np.ndarray), "y_train should be a list or a numpy array"
    assert isinstance(feature, int), "feature should be an int"

    median = find_median(x_train, feature)
    partitions = make_partitions(x_train, y_train, median, feature)
    x_train_left, y_train_left, x_train_right, y_train_right = partitions

    p1 = len([x for x in y_train_left if x])
    n1 = len(y_train_left) - p1

    p2 = len([x for x in y_train_right if x])
    n2 = len(y_train_right) - p2

    # if some partition is empty, the remainder is infinite because it means that the feature is useless
    if p1+n1 == 0 or p2+n2 == 0:
        return float("inf")

    h1 = h_entropy(p1, n1)
    r1 = h1 * (len(x_train_left) / len(x_train))

    h2 = h_entropy(p2, n2)
    r2 = h2 * (len(x_train_right) / len(x_train))

    return r1 + r2


def min_remainder(x_train: list, y_train: list, features: list[int]) -> int:
    """
    This function is used to find the feature that minimizes the remainder.\n
    In theory we should maximize the information gain, but since the first part
    of the gain formula it's the same for all the features, we can directly calculate
    the minimum reminder
    :param x_train: x train set
    :param y_train: y train set
    :param features: iterable list of features
    :return: the feature that minimizes the remainder
    """
    assert isinstance(x_train, list) or isinstance(x_train, np.ndarray), "x_train should be a list or a numpy array"
    assert isinstance(y_train, list) or isinstance(y_train, np.ndarray), "y_train should be a list or a numpy array"
    assert isinstance(features, list), "feature should be an int"

    if len(features) == 0:
        raise ValueError("features should have size > 0")

    return min([(f, remainder(x_train, y_train, f)) for f in features], key=lambda x: x[1])[0]


class MyDecisionTree:
    """
    This class is used to train and stores a binary decision tree classifier that splits the data of every feature
    using their median.\n
    Use the "fit" method to train the classifier and the "predict" method to predict the class of a new sample.
    """
    def __init__(self):
        self.tree = None

    def fit(self, x_train: list, y_train: list) -> None:
        """
        This method is used to train the classifier.

        :param x_train: a matrix where each row is a sample and each column is a feature
        :param y_train: a list of boolean values where each value is the class of the corresponding sample
        """
        if len(x_train) != len(y_train):
            raise ValueError("x_train e y_train should have the same size")
        if len(x_train) == 0:
            raise ValueError("x_train e y_train should not be empty")
        if len(x_train[0]) == 0:
            raise ValueError("x_train should contains non empty list")

        # TODO: maybe a set is better than a list for "features"
        features = list(range(len(x_train[0])))

        self.tree = self._make_tree(x_train, y_train, features, y_train)

    def _make_tree(self, x_train: list, y_train: list, features: list, y_train_parent: list) -> [TreeNode, bool]:
        if len(x_train) == 0:
            return plurality_value(y_train_parent)

        positive = len([y for y in y_train if y])
        if positive == len(y_train):
            return True
        if positive == 0:
            return False

        if len(features) == 0:
            return plurality_value(y_train)

        f = min_remainder(x_train, y_train, features)
        median = find_median(x_train, f)

        partitions = make_partitions(x_train, y_train, median, f)
        x_train_left, y_train_left, x_train_right, y_train_right = partitions

        features.remove(f)
        left = self._make_tree(x_train_left, y_train_left, features, y_train)

        right = self._make_tree(x_train_right, y_train_right, features, y_train)
        features.append(f)

        return TreeNode(f, median, left, right)

    def predict(self, x_test: list) -> list[bool]:
        assert isinstance(x_test, list) or isinstance(x_test, np.ndarray), "x_train_row should be a list or a numpy array"
        return [self.predict_single(x) for x in x_test]

    def predict_single(self, x_train_row: list) -> bool:
        assert isinstance(x_train_row, list) or isinstance(x_train_row, np.ndarray), "x_train_row should be a list or a numpy array"

        if self.tree is None:
            raise ValueError('You should train the model before using it; use the "fit" method')

        node = self.tree
        while not isinstance(node, bool):
            if x_train_row[node.feature] <= node.value:
                node = node.left
            else:
                node = node.right
        return node
