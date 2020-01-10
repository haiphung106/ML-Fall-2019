"""
Created by haiphung106
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import matplotlib.gridspec as gridspec
"""
Define input
"""
height = 195
width = 231
img_pixel = height * width

num_subjects = 15
num_labels = 11
num_image = num_subjects * num_labels

def read_pgm(filename):
    root = 'subject'
    labels = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses' ,
            'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']

    train_img = np.zeros((img_pixel, 135))
    test_img = np.zeros((img_pixel, 30))
    num_train_img = 0
    num_test_img = 0

    for sub_idx in range(num_subjects):
        test_idx = np.random.randint(0, num_labels, 2)
        if test_idx[0] == test_idx[1]:
            test_idx[1] = (test_idx[1] + 1) % num_labels

        for i in range(num_labels):
            img_filename = filename + root + ('%02d' % (sub_idx +1)) + '.' + labels[i] +'.pgm'
            img = Image.open(img_filename)
            img = img.resize((width, height), Image.NEAREST)

            if i in test_idx:
                test_img[:, num_test_img] = np.array(img).reshape(-1, )
                num_test_img += 1
            else:
                train_img[:, num_train_img] = np.array(img).reshape(-1, )
                num_train_img += 1

    return train_img, test_img


def lda(data):
    # Compute mean
    mean = np.zeros([5, img_pixel])
    for i in range(0, data.shape[1]):
        for j in range(0, data.shape[1]):
            mean[int(i / 1000)][j] += data[i][j]

        for i in range(0, 5):
            for j in range(0, data.shape[1]):
                mean[i][j] /= 1000

        all_mean = np.mean(data, axis=0)

    # Compute the within class scatter matrix
    within_class_matrix = np.zeros([data.shape[1], data.shape[1]])
    for i in range(0, data.shape[0]):
        temp = np.subtract(data[i], mean[int(i/ 1000)]).reshape(data.shape[1], 1)
        within_class_matrix += np.matmul(temp, temp.T)

    # Compute the between class scatter matrix
    between_class_matrix = np.zeros([data.shape[1], data.shapep[1]])
    for i in range(0, 5):
        temp2 = np.subtract(mean[i], all_mean).reshape(data.shape[1], 1)
        between_class_matrix += np.matmul(temp2, temp2.T)
    between_class_matrix *= 1000

    eigenValues, eigenVectors = np.linalg.eig(np.matmul(np.linalg.pinv(within_class_matrix), between_class_matrix))
    largestEigenVectors_idx = np.argsort(eigenValues)[::-1]

    return eigenVectors[:, largestEigenVectors_idx][:,:2]


if __name__ == '__mean__':
    train_img, test_img = read_pgm('Yale_Face_Database/all/')
    LDA = lda(train_img)
