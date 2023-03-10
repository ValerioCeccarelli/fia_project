import math
import numpy as np


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


def boolean_mode(boolean_list: [bool]) -> bool:
    """
    return the most frequently repeated boolean element

    :param boolean_list: a list containing boolean value
    """
    assert isinstance(boolean_list, list) or isinstance(boolean_list, np.ndarray), "boolean_list should be a list object"
    assert len(boolean_list) > 0, "boolean_list should not be empty"

    positive = len([y for y in boolean_list if y])
    if positive >= len(boolean_list) / 2:
        return True

    # TODO: the pseudocode says to choose random if there is a tie
    return False
