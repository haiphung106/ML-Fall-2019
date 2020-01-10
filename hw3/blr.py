# import data
# import numpy as np
# import random
#
#
# def _estimator(mean_old, SSDM_old, variance_old, point, count):
#     mean = mean_old + (point - mean_old) / count
#     SSDM = SSDM_old + (point - mean_old) * (point - mean)
#     variance = SSDM / (count - 1)
#     return (mean, SSDM, variance, count + 1)
#
#
# def estimator(mean, variance):
#     mean_est = data.gaussianGenerator(mean, variance)
#     SSDM = 0
#     variance_est = 0
#     count = 2
#     while (True):
#         point = data.gaussianGenerator(mean, variance)
#         (mean_est_new, SSDM, variance_est_new, count) = _estimator(mean_est, SSDM, variance_est, point, count)
#         print('-' * 20)
#         print('Current Point: {}'.format(point))
#         print('Estimate Mean: {}'.format(mean_est_new))
#         print('Estimate variance: {}'.format(variance_est_new))
#         if (abs(mean_est_new - mean_est) < 0.00001) & (abs(variance_est_new - variance_est) < 0.00001):
#             break
#         else:
#             mean_est = mean_est_new
#             variance_est = variance_est_new
#
#
# def BLR(basis, a, b, weight):
#     x = random.uniform(-10, 10)
#     y = data.polyBasisModel(basis, a, weight, x)
#     design_matrix = data.designMatrix(basis, x)
#     prior_mean = 0
#     prior_var_inv = b
#
#     # For the first point
#     posterier_var_inv = a * np.matmul(design_matrix.T, design_matrix) + b * np.eye(basis)
#     posterier_mean = a * np.matmul(np.linalg.inv(posterier_var_inv), design_matrix.T) * y
#     predictive_distribution_mean = np.matmul(design_matrix, posterier_mean)
#     predictive_distribution_var = 1 / a + \
#                                   np.matmul(np.matmul(design_matrix, np.linalg.inv(posterier_var_inv)), design_matrix.T)
#
#     print('-' * 20)
#     print('Current Data Point: ({}, {})'.format(x, y))
#     print('Posterier Parameter')
#     print('mean:\n{},\nco-variance: \n{}'.format(posterier_mean, np.linalg.inv(posterier_var_inv)))
#     print('Predictive Distribution')
#     print('mean:\n{},\nco-variance: \n{}'.format(predictive_distribution_mean, predictive_distribution_var))
#
#     # Online Learning
#     while (True):
#         x = random.uniform(-10, 10)
#         y = data.polyBasisModel(basis, a, weight, x)
#         design_matrix = data.designMatrix(basis, x)
#         prior_mean = posterier_mean.copy()
#         prior_var_inv = posterier_var_inv.copy()
#
#         posterier_var_inv = a * np.matmul(design_matrix.T, design_matrix) + prior_var_inv
#         posterier_mean = np.matmul(np.linalg.inv(posterier_var_inv), \
#                                    a * design_matrix.T * y + np.matmul(prior_var_inv, prior_mean))
#         predictive_distribution_mean = np.matmul(design_matrix, posterier_mean)
#         predictive_distribution_var = 1 / a + \
#                                       np.matmul(np.matmul(design_matrix, np.linalg.inv(posterier_var_inv)),
#                                                 design_matrix.T)
#         print('-' * 20)
#         print('Current Data Point: ({}, {})'.format(x, y))
#         print('Posterier Parameter')
#         print('mean:\n{},\nco-variance: \n{}'.format(posterier_mean, np.linalg.inv(posterier_var_inv)))
#         print('Predictive Distribution')
#         print('mean:\n{},\nco-variance: \n{}'.format(predictive_distribution_mean, predictive_distribution_var))
#
#         if (abs(np.sum(prior_mean - posterier_mean)) < 0.001) & \
#                 (abs(np.sum(prior_var_inv - posterier_var_inv)) < 0.001):
#             break
#
#
# BLR(2, 1, 1, [3, 2, 1])

# matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
np.random.seed(42)

beta = 25
def likelihood(x, t, w):
    likelihood_vals = stats.norm(W @ x, 1/np.sqrt(beta)).pdf(t)
    return np.squeeze(likelihood_vals)

# We will evaluate the likelihood function for different values of w
N = 50
w = np.linspace(-1, 1, N)
W = np.dstack(np.meshgrid(w, w))
# Observation
x = np.array([[1, 1.5]]).T
t = 1

# Plot likelihood function
plt.figure(figsize=(5,5))
plt.contourf(w, w, likelihood(x, 0, W), N)
plt.title('Likelihood function')
plt.xlabel('$w_1$')
plt.ylabel('$w_2$');

alpha = 2
def prior(w, mean=None, cov=None):
    if mean is None and cov is None:
        mean = np.array([[0, 0]]).T
        cov = 1/alpha * np.identity(2)
    prior_vals = stats.multivariate_normal(mean.flatten(), cov).pdf(w)
    return prior_vals

plt.figure(figsize=(5,5))
plt.contourf(w, w, prior(W), N);
plt.title('Prior distribution')
plt.xlabel('$w_1$')
plt.xlabel('$w_2$');


def sample_true_model():
    # True parameters
    a = np.array([-0.3, 0.5]).T
    x = np.array([[1, 2*np.random.random() - 1]]).T
    t = a.T @ x + np.random.normal(0, 0.2)
    return x, t

# Prior distribution parameters
m_0 = np.array([[0, 0]]).T
S_0 = 1/alpha * np.identity(2)

n_points = 5
obs_x = []
obs_t = []
for n in range(n_points):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)

    # Get new data point
    x, t = sample_true_model()

    # Plot likelihood
    plt.contourf(w, w, likelihood(x, t, W), N)
    plt.scatter(-0.3, 0.5, c='r', marker='x', label='True')
    plt.title('Likelihood')
    plt.legend()

    # Calculate posterior covariance and mean
    S_0_inv = np.linalg.inv(S_0)
    S_1 = np.linalg.inv(S_0_inv + beta * x @ x.T)
    m_1 = S_1 @ (S_0_inv @ m_0 + beta * t * x)

    # Plot posterior
    plt.subplot(1, 3, 2)
    plt.contourf(w, w, prior(W, mean=m_1, cov=S_1), N)
    plt.scatter(-0.3, 0.5, c='r', marker='x', label='True')
    plt.title('Posterior')
    plt.legend()

    # Plot observations
    obs_x.append(x[1])
    obs_t.append(t)
    plt.subplot(1, 3, 3)
    plt.scatter(obs_x, obs_t, label='Observations')
    # Add predictive mean
    n_pred = 100
    x_pred = np.vstack((np.ones(n_pred), np.linspace(-1, 1, n_pred)))
    pred_mean = m_1.T @ x_pred
    pred_var = np.sum(1/beta + x_pred.T @ S_1 * x_pred.T, axis=1)
    plt.plot(x_pred[1,:], pred_mean.T, 'r', label='Prediction')
    # Add predictive variance
    minus_var = pred_mean.flatten() - pred_var
    plus_var = pred_mean.flatten() + pred_var
    plt.fill_between(x_pred[1,:], minus_var, plus_var, alpha=0.1);
    plt.title('{:d} observations'.format(n + 1))
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.legend()

    plt.tight_layout()

    # For the next data point, the posterior will be a prior
    S_0 = S_1.copy()
    m_0 = m_1.copy()