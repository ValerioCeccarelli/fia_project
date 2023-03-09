from common.my_statistics import h_entropy, boolean_mode
import numpy as np


class TreeNode:
    def __init__(self, feature: int, value: float, left, right):
        assert isinstance(left, bool) or isinstance(left, TreeNode), f"left must be a TreeNode object or a boolean value but is: {type(left)}"
        assert isinstance(right, bool) or isinstance(right, TreeNode), "right must be a TreeNode object or a boolean value"
        assert isinstance(feature, int), "feature must be a int"
        assert isinstance(value, float), "value must be a float"

        # if feature < 0:
        #     raise ValueError(f"feature must be an int greater or equal than 0, currently is: {feature}")

        self.feature = feature
        self.value = value
        self.left = left
        self.right = right

    def __str__(self):
        return str([self.feature, self.value, self.left, self.right])

    def __repr__(self):
        return str(self)


def plurality_value(y_train: list[bool]) -> bool:
    return boolean_mode(y_train)


def find_median(x_train: list, feature: int):
    assert isinstance(x_train, list) or isinstance(x_train, np.ndarray), "x_train should be a list or a numpy array"
    assert isinstance(feature, int), "feature should be an int"

    # if feature < 0:
    #     raise ValueError(f"feature must be an int greater or equal than 0, currently is: {feature}")
    # if len(x_train) == 0:
    #     raise ValueError("x_train should have size > 0")
    # if feature >= len(x_train[0]):
    #     raise ValueError(f"feature should be a valid index, currently is: {feature} but x_train has {len(x_train[0])} features")

    x_train = sorted(x_train, key=lambda x: x[feature])
    if len(x_train) % 2 != 0:
        return x_train[len(x_train) // 2][feature]

    # TODO: check if this is correct
    return (x_train[len(x_train) // 2][feature] + x_train[len(x_train) // 2 - 1][feature]) / 2


def partizione(x_train: list, y_train: list[bool], median: float, feature: int):
    assert isinstance(x_train, list) or isinstance(x_train, np.ndarray), "x_train should be a list or a numpy array"
    assert isinstance(y_train, list) or isinstance(y_train, np.ndarray), "y_train should be a list or a numpy array"
    assert isinstance(median, float), "median should be a float"
    assert isinstance(feature, int), "feature should be an int"

    # if feature < 0:
    #     raise ValueError(f"feature must be an int greater or equal than 0, currently is: {feature}")
    # if len(x_train) == 0:
    #     raise ValueError("x_train should have size > 0")
    # if feature >= len(x_train[0]):
    #     raise ValueError(
    #         f"feature should be a valid index, currently is: {feature} but x_train has {len(x_train[0])} features")

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


def remainder(x_train: list, y_train: list[bool], feature: int):
    assert isinstance(x_train, list) or isinstance(x_train, np.ndarray), "x_train should be a list or a numpy array"
    assert isinstance(y_train, list) or isinstance(y_train, np.ndarray), "y_train should be a list or a numpy array"
    assert isinstance(feature, int), "feature should be an int"

    # if feature < 0:
    #     raise ValueError(f"feature must be an int greater or equal than 0, currently is: {feature}")
    # if len(x_train) == 0:
    #     raise ValueError("x_train should have size > 0")
    # if feature >= len(x_train[0]):
    #     raise ValueError(
    #         f"feature should be a valid index, currently is: {feature} but x_train has {len(x_train[0])} features")

    median = find_median(x_train, feature)
    partitions = partizione(x_train, y_train, median, feature)
    x_train_left, y_train_left, x_train_right, y_train_right = partitions

    p1 = len([x for x in y_train_left if x])
    n1 = len(y_train_left) - p1

    p2 = len([x for x in y_train_right if x])
    n2 = len(y_train_right) - p2

    # TODO: check if this is correct
    if p1 == 0 or p2 == 0:
        return 0

    h1 = h_entropy(p1, n1)
    r1 = h1 * (len(x_train_left) / len(x_train))

    h2 = h_entropy(p2, n2)
    r2 = h2 * (len(x_train_right) / len(x_train))

    return r1 + r2


def min_remainder(x_train, y_train, features):
    assert isinstance(x_train, list) or isinstance(x_train, np.ndarray), "x_train should be a list or a numpy array"
    assert isinstance(y_train, list) or isinstance(y_train, np.ndarray), "y_train should be a list or a numpy array"
    assert isinstance(features, list), "feature should be an int"

    # if len(features) == 0:
    #     raise ValueError("features should have size > 0")

    min_r = 1
    min_f = None
    for f in features:
        r = remainder(x_train, y_train, f)
        if min_f is None:
            min_r = r
            min_f = f
        if r < min_r:
            min_r = r
            min_f = f

    return min_f


class MyDecisionTree:
    def __init__(self, x_train: list, y_train: list):
        if len(x_train) != len(y_train):
            raise ValueError("x_train e y_train should have the same size")
        if len(x_train) == 0:
            raise ValueError("x_train e y_train should not be empty")
        if len(x_train[0]) == 0:
            raise ValueError("x_train should contains non empty list")

        # TODO: maybe a set is better than a list for "features"
        features = list(range(len(x_train[0])))

        self.tree = self._crea_albero(x_train, y_train, features, y_train)

    def _crea_albero(self, x_train: list, y_train: list, features: list, y_train_parent: list) -> [TreeNode, bool]:
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

        partitions = partizione(x_train, y_train, median, f)
        x_train_left, y_train_left, x_train_right, y_train_right = partitions

        features = features.copy()
        features.remove(f)

        left = self._crea_albero(x_train_left, y_train_left, features, y_train)
        right = self._crea_albero(x_train_right, y_train_right, features, y_train)

        return TreeNode(f, median, left, right)

    def predict(self, x_test: list) -> list:
        return [self._predict(x) for x in x_test]

    def _predict(self, x: list) -> bool:
        node = self.tree
        while not isinstance(node, bool):
            if x[node.feature] <= node.value:
                node = node.left
            else:
                node = node.right
        return node
