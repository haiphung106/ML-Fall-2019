"""
Created by haiphung106
"""
import numpy as np
import matplotlib.pyplot as plt
from function import *

N, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2 = call_input()
X, y, d1x, d1y, d2x, d2y = call_data(N, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2)

# w = np.zeros((3, 1))
w = [[0.0], [0.0], [0.0]]
# w_new = np.zeros((3, 1))
w_new = [[0.0], [0.0], [0.0]]
X_transpose = transpose(X)

while (True):
    sigmoid_input = mul(X, w)
    partial_derivative = mul(X_transpose, minus_maxtrix(y, sigmoid(sigmoid_input)))
    w_new = update_function(w, partial_derivative)
    if average(w_new, w):
        break
    w = w_new
gradient_w = w

while (True):
    D = []
    for i in range(0, len(X)):
        temp = []
        for j in range(0, len(X)):
            if i == j:
                temp1 = -1.0 * (X[i][0] * w[0][0] + X[i][1] * w[1][0] + X[i][2] * w[2][0])
                temp2 = np.exp(temp1)
                # if np.isinf(temp2):
                #     temp2 = np.exp(700)
                temp.append(temp2 / ((1 + temp2) ** 2))
            else:
                temp.append(0.0)
        D.append(temp)
    H = mul(X_transpose, mul(D, X))
    sigmoid_input = mul(X, w)
    partial_derivative = mul(X_transpose, minus_maxtrix(y, sigmoid(sigmoid_input)))
    if determinat(H) == 0:
        # Gradient Descent
        w_new = update_function(w, partial_derivative)
    else:
        # Newton's Method:
        w_new = np.add(w, mul(inverse(H), partial_derivative))
    if average(w_new, w):
        break
    w = w_new


# Confusion matrix
print('Gradient descent:\n')
confusion_matrix = np.zeros((2, 2))
predict = sigmoid((mul(X, gradient_w)))
c1x = []
c1y = []
c2x = []
c2y = []
for i in range(0, len(predict)):
    if predict[i][0] < 0.5:
        c1x.append((X[i][0]))
        c1y.append(X[i][1])
    else:
        c2x.append(X[i][0])
        c2y.append(X[i][1])
for i in range(0, len(predict)):
    if y[i][0] == 0:
        if predict[i][0] < 0.5:
            confusion_matrix[0][0] += 1
        else:
            confusion_matrix[0][1] += 1
    if y[i][0] == 1:
        if predict[i][0] < 0.5:
            confusion_matrix[1][0] += 1
        else:
            confusion_matrix[1][1] += 1

print('w:')
for i in range(0, len(gradient_w)):
    print(gradient_w[i][0])
print('\nConfusion Matrix: ')
print('\t\t Predict cluster 1 Predict cluster 2')
print('Is cluster 1\t\t {}\t\t{}'.format(confusion_matrix[0][0], confusion_matrix[0][1]))
print('Is cluster 2\t\t {}\t\t{}\n'.format(confusion_matrix[1][0], confusion_matrix[1][1]))
print('Sensitivity (Successfully predict cluster 1): {}'.format(
    confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])))
print('Sensitivity (Successfully predict cluster 2): {}'.format(
    confusion_matrix[0][0] / (confusion_matrix[1][0] + confusion_matrix[1][1])))

plt.subplot(131)
plt.title('Ground truth')
plt.scatter(d1x, d1y, c = 'r')
plt.scatter(d2x, d2y, c = 'b')
plt.subplot(132)
plt.title('Gradient descent')

plt.scatter(c1x, c1y, c = 'r')
plt.scatter(c2x, c2y, c = 'b')

print('n\--------------------------------\nNewton`s method:\n')
confusion_matrix = [[0, 0], [0, 0]]
predict = mul(X, w_new)
c1x = []
c1y = []
c2x = []
c2y = []
for i in range(0, len(predict)):
    if predict[i][0] < 0.5:
        c1x.append(X[i][0])
        c1y.append(X[i][1])
    else:
        c2x.append(X[i][0])
        c2y.append(X[i][1])
for i in range(0, len(predict)):
    if y[i][0] == 0:
        if predict[i][0] < 0.5:
            confusion_matrix[0][0] += 1
        else:
            confusion_matrix[0][1] += 1
    if y[i][0] == 1:
        if predict[i][0] < 0.5:
            confusion_matrix[1][0] += 1
        else:
            confusion_matrix[1][1] += 1

print("w:")
for i in range(0, len(w_new)):
    print(w_new[i][0])
print("\nconfusion_matrix")
print("\t\t Predict cluster 1 Predict cluster 2")
print("Is cluster 1\t\t {}\t\t{}" .format(confusion_matrix[0][0], confusion_matrix[0][1]))
print("Is cluster 2\t\t {}\t\t{}\n" .format(confusion_matrix[1][0], confusion_matrix[1][1]))
print("Sensitivity (Successfully predict cluster 1): {}" .format(confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])))
print("Specificity (Successfully predict cluster 2): {}" .format(confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])))

plt.subplot(133)
plt.title("Newton's Method")
plt.scatter(c1x, c1y, c = 'r')
plt.scatter(c2x, c2y, c = 'b')

plt.tight_layout()
plt.show()