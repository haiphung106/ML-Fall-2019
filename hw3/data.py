import random
import numpy as np


def gaussianGenerator(mean, variance):
    """
    This function will return sampling point from a given gaussian distribution.
    """
    total = 0
    for _ in range(12):
        total += random.uniform(0, 1)
    z = total - 6
    return mean + z * (variance**0.5)


def polyBasisModel(basis, variance, weight, x):
    """
    This function will return a target point with a given linear polynomial model.
    """
    y = gaussianGenerator(0, variance)
    for i in range(basis):
        y += weight[i] * (x ** i)
    return y


def designMatrix(basis, x):
    design_matrix = []
    for i in range(basis):
        design_matrix.append(x ** i)
    return np.array(design_matrix).reshape(1, -1)