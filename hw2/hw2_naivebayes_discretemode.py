# Created by haiphung106

import numpy as np
from function import LoadingData, LoadingLabel

trainimage = LoadingData('train-images-idx3-ubyte')
trainlabel = LoadingLabel('train-labels-idx1-ubyte')
testimage = LoadingData('t10k-images-idx3-ubyte')
testlabel = LoadingLabel('t10k-labels-idx1-ubyte')
nlabel = 10
Pc = np.zeros(nlabel)
c = 0

# Prior
for c in range(10):
    count = 0
    for label in trainlabel:
        if label == c:
            count += 1
    Pc[c] = count / 60000  # number of time c apper /number of image

nrows = 28
ncols = 28
images = 60000

bin_images = np.copy(trainimage)
for i in range(images):
    for j in range(nrows):
        for k in range(ncols):
            bin_images[i][j][k] = bin_images[i][j][k] // 8

# Multinomial Method
# Likelihood
bins = 32
count = np.zeros((10, 28, 28, 32))
count_class = np.zeros(10)

for j in range(nrows):
    for k in range(ncols):
        for i in range(images):
            count[trainlabel[i]][j][k][bin_images[i][j][k]] += 1
            count_class[trainlabel[i]] += 1

# Posterior
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
                b = testimage[img_id][j][k] // 8
                likelihood += (np.log(count[c, j, k, b] + 1) - np.log(count_class[c] + 28 * 28 * 32))
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
    print('{}:'.format(c))
    for j in range(28):
        for k in range(28):
            zero_sum = 0
            one_sum = 0
            for b in range(32):
                if b < 16:
                    zero_sum += count[c][j][k][b]
                else:
                    one_sum += count[c][j][k][b]
            print(0 if zero_sum > one_sum else 1, end=' ')
        print('')
    print('')
print('Error rate = {}/10000'.format(error_rate))
