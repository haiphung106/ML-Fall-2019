"""
Created by haiphung106
"""
import numpy as np


def call_input():
    N = int(input('Number of data point: '))
    mx1 = float(input('mx1: '))
    my1 = float(input('my1: '))
    mx2 = float(input('mx2: '))
    my2 = float(input('my2: '))
    vx1 = float(input('vx1: '))
    vy1 = float(input('vy1: '))
    vx2 = float(input('vx2: '))
    vy2 = float(input('vy2: '))
    return N, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2


def gaussian_data_generator(mean, variance):
    temp = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
    return mean + np.sqrt(variance) * temp


def call_data(N, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2):
    d1x = []
    d1y = []
    d2x = []
    d2y = []
    X = []
    y = []
    for i in range(0, N):
        temp1 = []
        x1 = gaussian_data_generator(mx1, vx1)
        y1 = gaussian_data_generator(my1, vy1)
        d1x.append(x1)
        d1y.append(y1)
        temp1.append(x1)
        temp1.append(y1)
        temp1.append(1.0)
        X.append(temp1)
        y.append([0.0])

        temp2 = []
        x2 = gaussian_data_generator(mx2, vx2)
        y2 = gaussian_data_generator(my2, vy2)
        d2x.append(x2)
        d2y.append(y2)
        temp2.append(x2)
        temp2.append(y2)
        temp2.append(1.0)
        X.append(temp2)
        y.append([1.0])
    return X, y, d1x, d1y, d2x, d2y


def sigmoid(x):
    res = []
    for i in range(len(x)):
        res.append([1.0 / (1.0 + np.exp(-x[i][0]))])
    return res


def transpose(A):
    B = []
    for i in range(len(A[0])):
        temp = []
        for j in range(len(A)):
            temp.append(A[j][i])
        B.append(temp)
    return B


def identitymatrix(n):
    I = np.zeros((int(n), int(n)))
    for i in range(n):
        I[i, i] = 1.0
    return I


def mul(A, B):
    result = []
    for i in range(len(A)):
        temp = []
        for j in range(len(B[0])):
            temp.append(0)
        result.append(temp)
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


def designmatrix(n, x):
    design_matrix = []
    for i in range(n):
        design_matrix.append(x ** i)
    return np.array(design_matrix).reshape(-1, 1)


def inverse(X):
    n = len(X)
    L = [[0 * 0] * n for i in range(n)]
    U = [[0 * 0] * n for i in range(n)]
    for j in range(n):
        L[j][j] = 1
        for i in range(j + 1):
            s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = X[i][j] - s1

        for i in range(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (X[i][j] - s2) / U[j][j]

    inverse_L = np.zeros((n, n))
    inverse_matrix = np.zeros((n, n))
    I = identitymatrix(n)

    for k in range(n):
        for i in range(n):
            s1 = sum(L[i][j] * inverse_L[j][k] for j in range(i))
            inverse_L[i][k] = (I[i][k] - s1) / L[i][i]

    for k in range(n):
        for i in reversed(range(n)):
            s2 = sum(U[i][j] * inverse_matrix[j][k] for j in reversed(range(i + 1, n)))
            inverse_matrix[i][k] = (inverse_L[i][k] - s2) / U[i][i]

    return inverse_matrix


def first_order(A, B, C):
    return mul(mul(transpose(C), transpose(A)), A) - mul(transpose(B), A)


def minus_maxtrix(A, B):
    result = []
    for i in range(len(A)):
        temp = []
        for j in range(len(B[0])):
            temp.append(A[i][j] - B[i][j])
        result.append(temp)
    return result

def update_function(A, B):
    result = []
    rate = 0.01
    for i in range(len(A)):
        temp = []
        for j in range(len(A[0])):
            temp.append(A[i][j] + B[i][j] * rate)
        result.append(temp)
    return result


def average(A, B):
    temp = True
    for i in range(0, len(A)):
        if abs(A[i][0] - B[i][0]) > (abs(B[i][0]) * 0.05):
            temp = False
            break
    return temp


def determinat(A):
    return A[0][0] * (A[1][1] * A[2][2] - A[2][1] * A[1][2]) - A[0][1] * (
                A[1][0] * A[2][2] - A[1][2] * A[2][0] + A[0][2]) * (A[1][0] * A[2][1] - A[1][1] * A[2][0])


def LoadingData(fileName):
    data_type = np.dtype('int32').newbyteorder('>')
    data = np.fromfile(fileName, dtype='ubyte')
    X = data[4 * data_type.itemsize:].astype('float64').reshape(60000, 784)
    X = np.divide(X, 128).astype('int')
    # print(X)
    return X
def LoadingLabel(fileName):
    data_type = np.dtype('int32').newbyteorder('>')
    labels = np.fromfile(fileName, dtype='ubyte').astype('int')
    labels = labels[2 * data_type.itemsize:].reshape(60000)
    return labels
