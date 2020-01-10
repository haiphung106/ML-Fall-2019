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
height = 116
width = 98
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


def pca(data):
    mean = np.mean((data), axis=1).reshape(img_pixel, 1) #(11368, 1)
    # print(mean.shape)
    # centered = data - np.matlib.repmat(mean, len(data), 1) #(11368, 135)
    centered = data - mean #(11368, 135)
    # print(centered.shape)
    # covariance = np.dot(centered.T, centered) / len(data) #(135, 135)
    covariance = np.cov(centered.T) / data[1]
    K = 256
    # print(covariance.shape)
    # Sort of EigenVectors
    eigenValues, eigenVectors = np.linalg.eig(covariance) # (135, ) , (135, 135)
    eigenVectors = (centered @ eigenVectors).T
    largestEigenVector_idx = np.argsort(eigenValues)
    largestEigenVector_idx = eigenVectors[largestEigenVector_idx][::-1].astype(float)
    largestEigenVector_idx = np.true_divide(largestEigenVector_idx, np.linalg.norm(largestEigenVector_idx, ord = 2, axis=1).reshape(-1, 1))
    # print(largestEigenVector_idx.shape)

    # PCA = np.zeros((11368, 0))
    # for i in range(K):
    #     newCovariance = np.reshape(largestEigenVector_idx[:, largestEigenVector_idx[i]], (11368, 1))#(45045, 1)
    #     print('New covariance= {}'.format(newCovariance.shape))
    #     PCA = np.concatenate((PCA, newCovariance), axis=0)
    PCA = largestEigenVector_idx[:-K -1:-1] #(135, 11368)


    # PCA = (PCA - np.min(PCA, axis=0).reshape(-1, 1) * 255) / (np.max(PCA, axis=0).reshape(-1,1))
    print('PCA={}'.format(PCA.shape))
    # sortedEigenVectors = eigenValues[largestEigenVector_idx][1:]
    # plt.plot(sortedEigenVectors, 'r.')
    # plt.xlabel('Sort of EigenVectors')
    # plt.show()

    # Show EigenFace
    fig = plt.figure(figsize=(15, 15))
    for i in range(25):
        fig.add_subplot(5, 5, i + 1)
        eigenFace = np.reshape(PCA[i], (height, width))
        plt.axis('off')
        plt.title('EigenFace with K= {}'.format(K))
        plt.imshow(eigenFace, cmap='gray')
    plt.show()

    # Reconstruct Face
    W = centered.dot(PCA) #(11368, 135)
    print(W.shape)
    Reconstruct = W.dot(PCA) + mean
    fig = plt.figure(figsize=(10, 10))
    for i in range(10):
        fig.add_subplot(2, 5, i + 1)
        recon_img = np.reshape(Reconstruct[i], (height, width))
        plt.axis('off')
        plt.title('Reconstruct Face with K = {}'.format(K))
        plt.imshow(recon_img, cmap='gray')
    plt.show()



if __name__ == '__main__':
   train_img, test_img = read_pgm('Yale_Face_Database/all/')
   PCA = pca(train_img)