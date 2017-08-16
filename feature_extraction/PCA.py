import numpy as np
from scipy import linalg

def pca(A):
    # function [W_norm ,eigenvalue ,mean] = pca(A)
    # computes PCA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W_norm: D by K matrix whose columns are the principal components in decreasing order
    # eigenvalue: eigenvalues
    # mean: mean of columns of A

    (d, n) = np.shape(A)  # n: pixel #, m: sample #

    # mean-normalize
    mean = np.mean(A, axis=1).reshape([d,1])
    A_norm = A - mean # d x n

    # inner-product
    ATA = np.dot(A_norm.T, A_norm) # n x n
    [eigen_value, eigen_vector] = linalg.eig(ATA)  # [1 x n, n x n]

    # order eigenvectors
    order_index = np.argsort(eigen_value)
    order_index = order_index[::-1]
    eigen_value = eigen_value[order_index]
    eigen_vector = eigen_vector[:, order_index]

    # actual eigenvector
    W = np.dot(A_norm, eigen_vector) # d x n
    W_norm = W / linalg.norm(W, axis=0) # normalization

    return W_norm, eigen_value, mean