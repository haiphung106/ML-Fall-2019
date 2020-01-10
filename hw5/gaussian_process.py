"""
Created by haiphung106
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

"""
Read input
"""

x = []
y = []
with open('input.data') as file:
    for line in file:
        x.append(float(line.split()[0]))
        y.append(float(line.split()[1]))
train_x = np.array(x).reshape(-1, 1)
train_y = np.array(y).reshape(-1, 1)

beta = 5
noise = 1 / beta
test_x = np.arange(-60, 60, 1).reshape(-1, 1)

"""
Define kernel
"""

def kernel(x1, x2, length=1.0, sigma=1.0, alpha=1.0):
    return (sigma ** 2) * (1 + cdist(x1, x2, 'sqeuclidean') / (2 * alpha * length ** 2)) ** (-alpha)


"""
Gaussian Process Predict
"""

def GP_predict(test_x, train_x, train_y, length=1.0, sigma=1.0, alpha=1.0, noise=1 / 5):
    k = kernel(train_x, train_x, length, sigma, alpha) + noise * np.eye(len(train_x))
    k_train = kernel(train_x, test_x, length, sigma, alpha)
    mean_test_y = kernel(test_x, test_x, length, sigma, alpha) + noise
    k_inv = np.linalg.inv(k)
    mean = np.linalg.multi_dot([k_train.T, k_inv, train_y])
    covariance = mean_test_y - np.linalg.multi_dot([k_train.T, k_inv, k_train])
    return mean, covariance

"""
Gaussian Process Predict Plot
"""
def GP_plot(mean, covariance, test_x, train_x, train_y, fig_name):
    test_x = test_x.ravel()
    mean = mean.ravel()
    temp = 1.96 * np.sqrt(np.diag(covariance))
    plt.figure()
    plt.fill_between(test_x, mean + temp, mean - temp, alpha=0.25, color='r', label='95% confidence interval')
    plt.plot(test_x, mean, label='Mean', color='r')
    plt.plot(train_x, train_y, 'b.', label='Training data')
    plt.axis([-60, 60, -5, 5])
    plt.legend()
    plt.title(fig_name)
    plt.savefig(fig_name +'.png')
    # plt.show()

"""
Before de-noise
"""
GPdefault = True
if GPdefault == True:
    mean, covar = GP_predict(test_x, train_x, train_y, noise=noise)
    GP_plot(mean, covar, test_x, train_x=train_x, train_y=train_y,
            fig_name='Poster and prior distribution before de-noise')
"""
After de-noise
"""
GPoptimize = True
if GPoptimize == True:
    def NLL(test_x, train_x, train_y):
        length = test_x[0]
        sigma = test_x[1]
        alpha = test_x[2]
        k = kernel(train_x, train_x, length, sigma, alpha)
        temp1 = 0.5 * np.log(np.linalg.det(k))
        temp2 = 0.5 * np.linalg.multi_dot([train_y.T, np.linalg.inv(k), train_y])
        temp3 = 0.5 * train_x.shape[0] * np.log(2 * np.pi)
        return temp1 + temp2 + temp3

    optimize_result = minimize(fun=NLL, x0=np.array([1, 1, 1]), args=(train_x, train_y),
                               bounds=((1e-3, None), (1e-3, None), (1e-3, None)), method='L-BFGS-B')
    print(optimize_result.x)
    length_op, sigma_op, alpha_op = optimize_result.x
    mean, covar = GP_predict(test_x, train_x=train_x, train_y=train_y, length=length_op, sigma=sigma_op, alpha=alpha_op,
                             noise=noise)
    GP_plot(mean, covar, test_x, train_x=train_x, train_y=train_y,
            fig_name='Poster and prior distribution after de-noise')
#
