""" 
Created by haiphung106 
"""

from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import random

"""
Loading data
"""
data = []
n = 982
for i in range(n):
    filename = "four_dataset/four" + str(i) + ".jpg"
    img = Image.open(filename)
    data.append(np.reshape(img, (-1, 1)))

data = np.squeeze(data)
num = data.shape[0]

"""
Calculate mean, covariance
"""
mean = np.mean(data, axis=0)
centered = data - np.matlib.repmat(mean, num, 1)
covariance = np.dot(centered.T, centered) / (num - 1)
"""
Do PCA
"""
K = 256
D = 28 * 28
U = np.zeros((D, K))
U_ = np.zeros((D, D - K))
"""
Sort of EigenVectors
"""
eigenValues, eigenVectors = np.linalg.eig(covariance)
largestEigenVectorIndex = np.argsort(eigenValues.real)
largestEigenVectorIndex = largestEigenVectorIndex[::-1]
PCA = np.zeros((D, 0))
for i in range(K):
    newCov = np.reshape(eigenVectors[:, largestEigenVectorIndex[i]], (D, 1))
    PCA = np.concatenate((PCA, newCov), axis=1)

sortedEigenVector = eigenValues[largestEigenVectorIndex][100:]
plt.plot(sortedEigenVector, 'b.')
plt.xlabel('Sort of EigenVector')

"""
Make eigennum
"""
isDrawPCA = True
if (isDrawPCA == True):
    plt.figure(figsize=(1, 1))
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    for i in range(1):
        for j in range(1):
            eigenNum = np.reshape(PCA[:, (i * 1 + j)], (28, 28))
            ax = plt.subplot(gs1[i * 1 + j])
            ax.axis('off')
            ax.imshow(eigenNum.real, cmap='gray')
    plt.savefig(r'F:\My Drive\NCTU\Ph.D\Fall 2019\Machine Learning for Signal processing\hw\hw3\eigenfour.jpg',
                bbox_inches='tight')
    plt.clf()

"""
Reconstruct Num
"""
W = centered.dot(PCA)
reconstruct = W.dot(PCA.T) + mean
print('PCA Shape: {}'.format(PCA.shape))
print(reconstruct.shape)
isDrawReconstructedNum = True
if (isDrawReconstructedNum):
    nNum = 1
    idx = 0
    plt.figure(figsize=(1, 1))
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    for i in range(1):
        for j in range(1):
            ReconstructedImg = np.reshape(reconstruct[i * 1 + j], (28, 28))
            ax = plt.subplot(gs1[i * 1 + j])
            ax.axis('off')
            ax.imshow(ReconstructedImg.real, cmap='gray')
    plt.savefig(
        r'F:\My Drive\NCTU\Ph.D\Fall 2019\Machine Learning for Signal processing\hw\hw3\Reconstructed_four.jpg',
        bbox_inches='tight')
    plt.clf()

    # Original Num
    for i in range(1):
        for j in range(1):
            numImg = np.reshape(data[i * 1 + j], (28, 28))
            ax = plt.subplot(gs1[i * 1 + j])
            ax.axis('off')
            ax.imshow(numImg.real, cmap='gray')
    plt.savefig(r'F:\My Drive\NCTU\Ph.D\Fall 2019\Machine Learning for Signal processing\hw\hw3\Original_four.jpg',
                bbox_inches='tight')
    plt.clf()
plt.show()