# Created by haiphung106
import numpy as np


def call_input():
    print('a = ', end='')
    a = int(input())
    print('b = ', end='')
    b = int(input())
    # standard_deviation = np.sqrt(variance)
    print('n = ', end='')
    n = int(input())
    # print('w = ', end='')
    # w = range(int(input()))
    return a, b, n

def gaussian_data_generator(mean, variance):
    temp = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
    return mean + np.sqrt(variance) * temp

def poly_linear_model_dara_generator(n, a, w):
    y = gaussian_data_generator(0, a)
    x = np.random.uniform(-10, 10)
    for i in range(n):
        y += w[i] * (x**i)
    return y

def transpose(A):
    row = A.shape[0]
    col = A.shape[1]
    B = np.zeros((col, row))
    for i in range(row):
        for j in range(col):
            B[j][i] = A[i][j]
    return B

def identitymatrix(n):
    I = np.zeros((int(n), int(n)))
    for i in range(n):
        I[i, i] = 1.0
    return I


def mul(A, B):
    result = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i][j] += A[i][k] * B[k][j]
    return result

def design_matrix(n, x):
    design_matrix = []
    for i in range(n):
        design_matrix.append(x ** i)
    return np.array(design_matrix).reshape(1, -1)


