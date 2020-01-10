# Created by haiphung106

import numpy as np
from function import LoadingData, LoadingLabel

trainimage = LoadingData('train-images-idx3-ubyte')
trainlabel = LoadingLabel('train-labels-idx1-ubyte')
testimage = LoadingData('t10k-images-idx3-ubyte')
testlabel = LoadingLabel('t10k-labels-idx1-ubyte')
nlabel = 10
Pc = np.zeros(nlabel)
count = np.zeros(10)

# Prior
for c in range(10):
    count[c] = 0
    for label in trainlabel:
        if label == c:
            count[c] += 1
    Pc[c] = count[c] / 60000  # number of time c apper /number of image

nrows = 28
ncols = 28
images = 60000

expected_value = np.zeros((10, 28, 28))
var = np.zeros((10, 28, 28))

# Multinomial Method
# u, variance

for j in range(nrows):
    for k in range(ncols):
        for i in range(images):
            expected_value[trainlabel[i]][j][k] += trainimage[i][j][k]
        for c in range(10):
            expected_value[c][j][k] /= count[c]
        for i in range(images):
            var[trainlabel[i]][j][k] += (trainimage[i][j][k] - expected_value[trainlabel[i]][j][k]) ** 2
        for c in range(10):
            var[c][j][k] /= count[c]
            if var[c][j][k] == 0:
                var[c][j][k] = 2000

# Posterior
# Likelihood
error_rate = 0
for img_id in range(10000):
    selected_class = -1
    maximum_prob = -10000000000
    result = np.zeros(10)
    total = 0
    print('Posterior (in log scale):')
    for c in range(10):
        likelihood = 0
        for j in range(nrows):
            for k in range(ncols):
                likelihood += (-((testimage[img_id][j][k] - expected_value[c][j][k]) ** 2) / (
                            2 * var[c][j][k])) - 0.5 * np.log(2 * np.pi * var[c][j][k])
                # likelihood += np.log((1/(np.sqrt(2 * np.pi * var[c][j][k]))) - ((testimage[img_id][j][k] - expected_value[c][j][k])**2)/(2*var[c][j][k]))
        result[c] = np.log(Pc[c]) + likelihood
        total += result[c]
        if result[c] > maximum_prob:
            maximum_prob = result[c]
            selected_class = c

    if selected_class != testlabel[img_id]:
        error_rate += 1
    for c in range(10):
        nor_likelihood = result[c] / total
        print('{}: {}'.format(c, nor_likelihood))
    print('Prediction = {} Ans = {}'.format(selected_class, testlabel[img_id]))

for c in range(10):
    print('{}: '.format(c))
    for j in range(28):
        for k in range(28):
            print(0 if expected_value[c][j][k] < 128 else 1, end=' ')
        print('')
    print('')
print('Error rate = {}/10000'.format(error_rate))
