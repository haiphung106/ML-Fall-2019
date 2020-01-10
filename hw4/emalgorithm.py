"""
Created by haiphung106
"""

import numpy as np
from function import *

trainimage = LoadingData('train-images-idx3-ubyte')
trainlabel = LoadingLabel('train-labels-idx1-ubyte')
nImages = 60000

pi = np.full((10, 1), 0.1, dtype=np.float64)
mean = np.random.rand(784, 10).astype(np.float64)
prev_mean = np.zeros((784, 10), dtype=np.float64)
covariance = np.full((10, nImages), 0.1, dtype=np.float64)
iteration = 0

while (True):
    iteration += 1
    # E step
    for image_id in range(nImages):
        temp = np.zeros(10, dtype=np.float64)
        for c in range(10):
            temp1 = np.float64(1.0)
            for pixel_id in range(784):
                if trainimage[image_id][pixel_id]:
                    temp1 *= mean[pixel_id][c]
                else:
                    temp1 *= (1 - mean[pixel_id][c])
            temp[c] = pi[c][0] * temp1
        temp2 = np.sum(temp)
        if temp2 == 0:
            temp2 = 1
        for c in range(10):
            covariance[c][image_id] = temp[c] / temp2
    # M step
    N = np.sum(covariance, axis=1)
    for pixel_id in range(784):
        for c in range(10):
            temp = 0.0
            for image_id in range(nImages):
                temp += np.dot(covariance[c][image_id], trainimage[image_id][pixel_id])
            temp1 = N[c]
            if temp1 == 0:
                temp1 = 1
            mean[pixel_id][c] = (temp / temp1)

        for c in range(10):
            pi[c][0] = N[c] / nImages

    # check condition
    absolute = 0
    for pixel_id in range(784):
        for c in range(10):
            absolute += abs(mean[pixel_id][c] - prev_mean[pixel_id][c])
    # check label
    relation = []
    for c in range(10):
        print('Class: {}'.format(c))
        for pixel_id in range(784):
            if pixel_id % 28 == 0:
                print()
            if mean[pixel_id][c] >= 0.5:
                print('1', end='')
            else:
                print('0', end='')
        print()

    print('No. of iteration: {}, Difference: {}'.format(str(iteration), str(absolute)))

    if absolute < 10:
        break
    prev_mean = mean.copy()

# Confusion matrix
error = nImages
confusion_matrix = np.zeros(shape=(10, 2, 2), dtype=int)
# set label

cluster = [[], [], [], [], [], [], [], [], [], []]
image_cluster = []
for image_id in range(nImages):
    temp = np.zeros(10, dtype=np.float64)
    for c in range(10):
        temp1 = np.float(1.0)
        for pixel_id in range(784):
            if trainimage[image_id][pixel_id] == 1:
                temp1 *= mean[pixel_id][c]
            else:
                temp1 *= (1 - mean[pixel_id][c])
        temp[c] = pi[c][0] * temp1

    predict = np.argmax(temp)
    cluster[predict].append(image_id)
    image_cluster.append(predict)

cluster_label = np.zeros((10), dtype=np.int)
for cluster_id in range(10):
    count = np.zeros((10), dtype=np.int)
    for image_id in cluster[cluster_id]:
        ground_truth = trainlabel[image_id]
        count[ground_truth] += 1
    cluster_label[cluster_id] = np.argmax(count)

for c in range(10):
    print("\nLabeled class: " + str(cluster_label[c]))
    for pixel_id in range(784):
        if pixel_id % 28 == 0:
            print()
        if (mean[pixel_id][c] >= 0.5):
            print("1 ", end='')
        else:
            print("0 ", end='')

guess_true = 0
for image_id in range(nImages):
    predict = cluster_label[image_cluster[image_id]]
    ground_truth = trainlabel[image_id]
    if predict == ground_truth:
        guess_true += 1
        confusion_matrix[ground_truth][0][0] += 1
        for other_label in range(10):
            if other_label != ground_truth:
                confusion_matrix[other_label][1][1] += 1
    else:
        confusion_matrix[ground_truth][0][1] += 1
        confusion_matrix[predict][1][0] += 1
        for other_label in range(10):
            if other_label != ground_truth and other_label != predict:
                confusion_matrix[other_label][1][1] += 1

for i in range(10):
    print('\n-------------------------------------------------------\n')
    print('Confusion matrix {}: '.format(i))
    print('\t\t\t\t\tPredict number {}\t Predict not number {}'.format(i, i))
    print('Is number    {}\t\t\t{}\t\t\t{}'.format(i, confusion_matrix[i][0][0], confusion_matrix[i][0][1]))
    print('Isn`t number {}\t\t\t{}\t\t\t{}\n'.format(i, confusion_matrix[i][1][0], confusion_matrix[i][1][1]))
    print('Sensitivity (Successfully predict number {})\t: {}'.format(i, confusion_matrix[i][0][0] / (
                confusion_matrix[i][0][0] + confusion_matrix[i][0][1])))
    print('Sensitivity (Successfully predict number {})\t: {}'.format(i, confusion_matrix[i][1][1] / (
                confusion_matrix[i][1][0] + confusion_matrix[i][1][1])))

print('----------------------------------------------------------------\n')

print('\nTotal iteration to converge: {}'.format(iteration))
print('\nTotal error rate: {}'.format(float(nImages - guess_true) / nImages))
