"""
Created by haiphung106
"""
import csv
import numpy as np
from libsvm.python.svmutil import *


def read_input():
    with open('X_train.csv') as file:
        csv_reader = csv.reader(file, delimiter = ',')
        x_train = list(csv_reader)
        x_train = [[float(y) for y in x] for x in x_train]

    with open('Y_train.csv') as file:
        csv_reader = csv.reader(file, delimiter = ',')
        y_train_2nd = list(csv_reader)
        y_train = [y for x in y_train_2nd for y in x]
        y_train = [int(x) for x in y_train]

    with open('X_test.csv') as file:
        csv_reader = csv.reader(file, delimiter = ',')
        x_test = list(csv_reader)
        x_test = [[float(y) for y in x] for x in x_test]

    with open('Y_test.csv') as file:
        csv_reader = csv.reader(file, delimiter = ',')
        y_test_2nd = list(csv_reader)
        y_test = [y for x in y_test_2nd for y in x]
        y_test = [int(x) for x in y_test]

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def svm(x_train, y_train, x_test, y_test):
    kernel = ['Linear Kernel', 'Polynomial Kernel', 'RBF Kernel']
    for i in range(len(kernel)):
        print('Kernel Function: {}'.format(kernel[i]))
        prob = svm_problem(y_train, x_train)
        param = svm_parameter('-t {} -q'.format(i))
        model = svm_train(prob, param)
        return svm_predict(y_test, x_test, model)




