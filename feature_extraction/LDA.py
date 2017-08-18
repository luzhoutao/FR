import numpy as np
from scipy import linalg

def lda(A, labels):
    # function [ W, center, labels] = lda(A, Labels)
    # computes LDA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K(=C-1) matrix whose columns are the principal components in decreasing order
    # center: projection of each class
    # labels: unique label

    [n, m] = np.shape(A) # n is the dimension of data; m is number of samples
    mu = np.mean(A, axis=1)
    classes, unique_inverse, class_counts = np.unique(labels, return_inverse=True, return_counts=True)
    partition = [ A[:, unique_inverse==k] for k in range(len(classes)) ] # [mu1, mu2, ...]; mui = [a, b, c, ...].T
    class_mean = np.hstack([ np.sum(part, axis=1).reshape([n, 1]) / class_counts[k] for k, part in enumerate(partition)])
    print('class mean', class_mean)
    # compute within-class scatter matrix
    sw = np.zeros([n, n])
    for idx in range(len(classes)):
        sw += np.cov(partition[idx], rowvar=True) * (class_counts[idx]-1) # freedom degree
    print('sw', sw)

    # compute between-class scatter matrix (use mean of all data)
    # [WRONG] sb = np.cov(class_mean, rowvar=True) (not mean of class means)
    sb = np.zeros([n, n])
    for idx in range(len(classes)):
        dif = class_mean[:, idx] - mu
        sb += np.outer(dif, dif) * class_counts[idx]
    print('sb', sb) # TODO

    # matrix decomposition
    value, vector = linalg.eig(sb, sw)

    # ordering the eigenvalue and eigenvector and choose biggest k-1
    idx = np.argsort(value)[::-1][:len(classes)-1]
    ordered_value = value[idx]
    ordered_vector = vector[:, idx]
    ordered_vector = ordered_vector / linalg.norm(ordered_vector, axis=0)

    # compute projected centers
    centers = np.dot(ordered_vector.T, class_mean)

    return ordered_vector, centers, classes

'''
x = np.array([
    [1,2,3,4,5],
    [2,3,4,1,5],
    [1,4,2,3,5],
    [3,5,1,2,4],
    [3,2,1,4,5]
])
labels = [1, 2, 2, 1, 1]

x1 = np.array([
    [1.2, 2.1, 1.1, 1.1],
    [4.1, 3.1, 1.5, 1.3]
])

labels1 = [1, 1, 2, 2]
print(myLDA(x, labels))
print(lda(x, labels))
'''