"""
Created by haiphung106
"""

from PIL import Image
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


def read_input(filename):
    img = Image.open(filename)
    width, height = img.size
    pixel = np.array(img.getdata()).reshape((width * height, 3))  # color value
    coordinate = np.array([]).reshape(0, 2)  # coordinate of color

    for i in range(width):
        row = np.array(list(zip(np.full(width, i), np.arange(width)))).reshape(width, 2)
        coordinate = np.vstack([coordinate, row])

    return pixel, coordinate


def call_kernel():
    print('K = ', end=' ')
    K = int(input())
    return K


def RBF_kernel(color, coordinate):
    spatial_rbf = np.exp(-lambda_s * squareform(pdist(coordinate, 'sqeuclidean')))
    color_rbf = np.exp(-lambda_c * squareform(pdist(color, 'sqeuclidean')))
    return spatial_rbf * color_rbf


def initialization(data, initial_method):
    C = np.array(list(zip(np.random.randint(0, len(data), size=K), np.random.randint(0, len(data), size=K))))
    mean = np.random.randn(K, 2)
    if initial_method == 'random' or 'r':
        prev_clusters = np.random.randint(K, size=data.shape[0])
        return C, mean, prev_clusters

    elif initial_method == 'kmeans++' or 'k':
        prev_clusters = np.random.randint(low=0, high=data.shape[0], size=1)
        mean[0, :] = data[prev_clusters, :]
        for i in range(1, K):
            distance = np.zeros(data.shape[0], dtype=np.float)
            for i in range(0, data.shape[0]):
                distance[i] = np.linalg.norm(data[i, :] - mean[0, :])
            distance = distance / distance.sum()
            distance = np.random.choice(data.shape[0], 1, p=distance)
            mean[i, :] = data[distance, :]
        return C, mean, prev_clusters


def term3(gram_matrix, labels, dataidx, k_cluster):
    cluster_sum = 0
    kernel_sum = 0
    for i in range(labels.shape[0]):
        if labels[i] == k_cluster:
            cluster_sum += 1
    if cluster_sum == 0:
        cluster_sum = 1
    for i in range(gram_matrix.shape[0]):
        if labels[i] == k_cluster:
            kernel_sum += gram_matrix[dataidx][i]

    return (-2) * kernel_sum / cluster_sum


def term2(gram_matrix, clusters):
    cluster_sum = np.zeros(K, dtype=np.int)
    kernel_sum = np.zeros(K, dtype=np.float)
    for i in range(clusters.shape[0]):
        cluster_sum[clusters[i]] += 1
    for cluster in range(K):
        for p in range(gram_matrix.shape[0]):
            for q in range(gram_matrix.shape[0]):
                if clusters[p] == cluster and clusters[q] == cluster:
                    kernel_sum[cluster] += gram_matrix[p][q]
    for cluster in range(K):
        if cluster_sum[cluster] == 0:
            cluster_sum[cluster] = 1
        kernel_sum[cluster] /= (cluster_sum[cluster] ** 2)

    return kernel_sum


def clustering(data, gram_matrix, mean, clusters):
    current_labels = np.zeros(data.shape[0], dtype=np.int)
    temp = term2(gram_matrix, clusters)
    for data_id in range(data.shape[0]):
        distance = np.zeros(K, dtype=np.float32)
        for cluster in range(K):
            distance[cluster] = term3(gram_matrix, clusters, data_id, cluster) + temp[
                cluster]
        current_labels[data_id] = np.argmin(distance)

    return current_labels


def accuracy(clusters, prev_clusters):
    error = 0
    for i in range(clusters.shape[0]):
        error += np.absolute(clusters[i] - prev_clusters[i])

    return error


def cluster_visualization(data, savename, iteration, clusters, initial_method):
    colors = ['red', 'green', 'blue', 'yellow']
    K = len(clusters)
    for i in range(len(data)):
        data_point = data[i]
        plt.scatter(data_point[0], data_point[1], color=colors[clusters][i][0])
    plt.savefig(savename + '_' + initial_method + '_' + str(K) + '_' + str(iteration) + '.png')


def k_means(filename, savename, data):
    method = ['random', 'kmeans++']
    for initial_method in method:
        C, mean, labels = initialization(data, initial_method)
        gram_matrix = RBF_kernel(data[0], data[1])
        iter = 0
        error = -10000
        prev_error = -10001
        print('Method: {}'.format(initial_method))
        print("mean = {}".format(mean))

        while True:
            if iter <= epochs:
                iter += 1
                print("iteration = {}".format(iter))
                prev_labels = labels
                cluster_visualization(data, savename, iter, labels, initial_method)
                labels = clustering(data, gram_matrix, mean, labels)
                error = accuracy(labels, prev_labels)
                print("error = {}".format(error))

                if error == prev_error:
                    break
                prev_error = error
            else:
                break
        return labels


def draw_eigenspace(data, savename, iteration, clusters, initial_method):
    K = len(clusters)
    colors = ['red', 'green', 'blue', 'yellow']
    for cluster in range(K):
        plt.clf()
        for i in range(len(data)):
            if clusters[i] == cluster:
                data_point = data[i]
                plt.scatter(data_point[0], data_point[1], color=colors[clusters][i][0])
        plt.title('Spectral-Clustering in Eigen-Space')
        plt.savefig(savename + '_' + initial_method + '_' + str(K) + '_' + str(iteration) + '.png')


def ratio_cut(data):
    W = RBF_kernel(data[0], data[1])
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    eigenValue, eigenVector = np.linalg.eig(L)
    sorted_id = np.argsort(eigenValue)[1: K + 1]
    U = eigenVector[:, sorted_id].astype(float)

    return U


def normalize_cut(data):
    W = RBF_kernel(data[0], data[1])
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    L = np.matmul(np.linalg.inv(D), L)
    eigenValue, eigenVector = np.linalg.eig(L)
    sorted_id = np.argsort(eigenValue)[1: K + 1]
    U = eigenVector[:, sorted_id].astype(float)

    norm_value = np.sum(U, axis=1)
    U = U / norm_value[:, None]
    return U


def spectral_cluster(data, filename, savename):
    method = ['random', 'kmeans++']
    for initial_method in method:
        C, mean, labels = initialization(data, initial_method)
        gram_matrix = RBF_kernel(data[0], data[1])
        iter = 0
        error = -10000
        prev_error = -10001
        print('Method: {}'.format(initial_method))
        print("mean = {}".format(mean))

        while True:
            if iter <= epochs:
                iter += 1
                print("iteration = {}".format(iter))
                prev_labels = labels
                cluster_visualization(data, savename, iter, labels, initial_method)
                labels = clustering(data, gram_matrix, mean, labels)
                error = accuracy(labels, prev_labels)
                print("error = {}".format(error))

                if error == prev_error:
                    break
                prev_error = error
            else:
                break

        draw_eigenspace(data, filename, savename, iter, labels, initial_method)


if __name__ == '__main__':


    width = 100
    height = 100
    epochs = 20
    K = call_kernel()
    lambda_c = 1 / (width * height)
    lambda_s = 1 / (width * height)

    print('Normalizing Cut is beginning here')
    print('Start with image 1')
    filename = 'image1.png'
    savename = 'visualization/image1'
    savename1 = 'spectra/image1_normalize'
    data = read_input(filename='image1.png')
    k_means(filename, savename, data)
    U = normalize_cut(data)
    spectral_cluster(filename, savename1, U)

    print('Continue with image 2')
    filename = 'image2.png'
    savename = 'visualization/image2'
    savename1 = 'spectra/image2_normalize'
    data = read_input(filename)
    k_means(filename, savename, data)
    U = normalize_cut(data)
    spectral_cluster(filename, savename1, U)

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')


    print('Ratio Cut is beginning here')
    print('Start with image 1')
    filename = 'image1.png'
    savename = 'visualization/image1'
    savename1 = 'spectra/image1_ratio'
    data = read_input(filename='image1.png')
    k_means(filename, savename, data)
    U = ratio_cut(data)
    spectral_cluster(filename, savename1, U)

    print('Continue with image 2')
    filename = 'image2.png'
    savename = 'visualization/image2'
    savename1 = 'spectra/image2_ratio'
    data = read_input(filename)
    k_means(filename, savename, data)
    U = ratio_cut(data)
    spectral_cluster(filename, savename1, U)
