"""
Created by haiphung106
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from libsvm.python.svmutil import *
from libsvm.python.svm import *
from function import *

"""
Read input
"""

with open('X_train.csv') as file:
    csv_reader = csv.reader(file, delimiter=',')
    x_train = list(csv_reader)
    x_train = [[float(y) for y in x] for x in x_train]

with open('Y_train.csv') as file:
    csv_reader = csv.reader(file, delimiter=',')
    y_train_2nd = list(csv_reader)
    y_train = [y for x in y_train_2nd for y in x]
    y_train = [int(x) for x in y_train]

with open('X_test.csv') as file:
    csv_reader = csv.reader(file, delimiter=',')
    x_test = list(csv_reader)
    x_test = [[float(y) for y in x] for x in x_test]

with open('Y_test.csv') as file:
    csv_reader = csv.reader(file, delimiter=',')
    y_test_2nd = list(csv_reader)
    y_test = [y for x in y_test_2nd for y in x]
    y_test = [int(x) for x in y_test]

"""
Compare the different between three kernel function
"""
Comparemode = False
if Comparemode == True:
    kernel = ['Linear Kernel', 'Polynomial Kernel', 'RBF Kernel']
    for i in range(len(kernel)):
        print('Kernel Function: {}'.format(kernel[i]))
        prob = svm_problem(y_train, x_train)
        param = svm_parameter('-t {} -q'.format(i))
        model = svm_train(prob, param)
        model_predict = svm_predict(y_test, x_test, model)

"""
Grid Research using C-SVC for Cross validation
"""

#Linear kernel with 3 - fold cross validation
Linearmodel = True
if Linearmodel == True:
    cost = -1
    gamma = -1
    degree = -1
    param_label = []
    gridsearch = -1
    for log2c in range(-8, 1, 1):
        param = svm_parameter('-q -t 0 -v 3 -c {}'.format(2 ** log2c))
        prob = svm_problem(y_train, x_train)
        model = svm_train(prob, param)
        param_label.append([2 ** log2c, model])
        if model > gridsearch:
            gridsearch = model
            cost = 2 ** log2c
    for label in param_label:
        print(label)
    # Test accuracy
    param = svm_parameter('-q -t 0 -c {}'.format(cost))
    prob = svm_problem(y_train, x_train)
    model = svm_train(prob, param)
    model_predict = svm_predict(y_test, x_test, model)

# Polynomial kernel with 3 - fold cross validation
Polynomialmodel = False
if Polynomialmodel == True:
    cost = -1
    gamma = -1
    degree = -1
    param_label = []
    gridsearch = -1
    for d in range(1, 5, 1):
        for log2g in range(-2, 2, 1):
            for log2c in range(-8, 1, 1):
                param = svm_parameter('-q -t 1 -v 3 -c {} -g {} -d {}'.format(2 ** log2c, 2 ** log2g, d))
                prob = svm_problem(y_train, x_train)
                model = svm_train(prob, param)
                param_label.append([2 ** log2c, 2 ** log2g, model])
                if model > gridsearch:
                    gridsearch = model
                    cost = 2 ** log2c
                    degree = d
                    gamma = 2 ** log2g
    for label in param_label:
        print(label)
    # Test accuracy
    param = svm_parameter('-q -t 1 -c {} -g {} -d {}'.format(cost, gamma, degree))
    prob = svm_problem(y_train, x_train)
    model = svm_train(prob, param)
    model_predict = svm_predict(y_test, x_test, model)

# RBF kernel with 3 - fold cross validation
RBFmodel = False
if RBFmodel == True:
    cost = -1
    gamma = -1
    degree = -1
    param_label = []
    gridsearch = -1

    for log2g in range(-5, 2, 1):
        for log2c in range(-3, 9, 1):
            param = svm_parameter('-q -t 2 -v 3 -c {} -g {}'.format(2 ** log2c, 2 ** log2g))
            prob = svm_problem(y_train, x_train)
            model = svm_train(prob, param)
            param_label.append([2 ** log2c, 2 ** log2g, model])
            if model > gridsearch:
                gridsearch = model
                cost = 2 ** log2c
                gamma = 2 ** log2g
    for label in param_label:
        print(label)
    # Test accuracy
    param = svm_parameter('-q -t 2 -c {} -g {}'.format(cost, gamma))
    prob = svm_problem(y_train, x_train)
    model = svm_train(prob, param)
    model_predict = svm_predict(y_test, x_test, model)

"""
Merger Linear and RBF kernel
"""
Mergemode = False
if Mergemode == True:
    #Compute kernel

    #Train
    negative_gamma = -1 / 4
    train_linear_kernel = np.matmul(x_train, np.transpose(x_train))
    train_rbf_kernel = squareform(np.exp(negative_gamma * pdist(x_train, 'sqeuclidean')))
    x_train_kernel = np.hstack((np.arange(1, 5001).reshape((5000, 1)), np.add(train_linear_kernel, train_rbf_kernel)))

    #Test
    test_linear_kernel = np.matmul(x_test, np.transpose(x_train))
    test_rbf_kernel = np.exp(negative_gamma * cdist(x_test, x_train, 'sqeuclidean'))
    x_test_kernel = np.hstack((np.arange(1, 2501).reshape((2500, 1)), np.add(test_linear_kernel, test_rbf_kernel)))

    #Predict new kernel
    prob = svm_problem(y_train, x_train_kernel, isKernel=True)
    param = svm_parameter('-t 4 -q')
    model = svm_train(prob, param)
    model_predict = svm_predict(y_test, x_test_kernel, model)

