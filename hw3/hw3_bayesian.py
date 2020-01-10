# Created by haiphung106
from function import poly_linear_model_dara_generator, call_input, mul, transpose, design_matrix
import numpy as np
import matplotlib.pyplot as plt


a, b , n = call_input()
w = np.array([1, 2, 3, 4])
x = np.random.uniform(-1.0, 1.0)
y = poly_linear_model_dara_generator(n, a, w)


design_matrix = design_matrix(n, x)
"""
1st data point
"""
# prior_mean = 0
# prior_variance_inverse = b

posterior_variance_inverse = a * mul(transpose(design_matrix), design_matrix) + b * np.eye(n)
posterior_mean = a * mul(posterior_variance_inverse, transpose(design_matrix)) * y
predictive_distribution_mean = mul(design_matrix, posterior_mean)
predictive_distribution_variance = 1 / a + mul(mul(design_matrix, posterior_variance_inverse), transpose(design_matrix))

print('Add data point ({}, {}):'.format(x, y))
print('Posterior mean: ')
print('{}'.format(posterior_mean))
print('Posterior variance:')
print('{}'.format(np.linalg.inv(posterior_variance_inverse)))
print('Predictive distribution ~ N',end='')
print('{}'.format(predictive_distribution_mean), '{}'.format(predictive_distribution_variance))

while(True):
    x = np.random.uniform(-1.0, 1.0)
    y = poly_linear_model_dara_generator(n, a, w)
    design_matrix = design_matrix(n, x)
    prior_mean = np.copy(posterior_mean)
    prior_variance_inverse = np.copy(posterior_variance_inverse)

    posterior_variance_inverse = a * mul(transpose(design_matrix), design_matrix) + prior_variance_inverse
    posterior_mean = mul(posterior_variance_inverse, a * transpose(design_matrix) * y + mul(prior_variance_inverse, prior_mean))
    predictive_distribution_mean = mul(design_matrix, posterior_mean)
    predictive_distribution_variance = 1 / a + mul(mul(design_matrix, posterior_variance_inverse), transpose(design_matrix))
    print('Add data point ({}, {}):'.format(x, y))
    print('Posterior mean: ')
    print('{}'.format(posterior_mean))
    print('Posterior variance:')
    print('{}'.format(np.linalg.inv(posterior_variance_inverse)))
    print('Predictive distribution ~ N', end='')
    print('{}'.format(predictive_distribution_mean), '{}'.format(predictive_distribution_variance))

    if abs(prior_mean - posterior_mean) <0.001 and abs(prior_variance_inverse - posterior_variance_inverse) < 0.001:
        break








