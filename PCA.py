# PCA Algorithm
# returns the W matrix which transforms to the basis with diagonal Covariance Matrix

import numpy as np

def PCA(M):
# Centering the data
    M_centered = M - np.mean(M, axis=0)
    C_X = np.cov(M_centered)
    eigenvalues, eigenvectors = np.linalg.eig(C_X)
    ncols = np.argsort(eigenvalues)[::-1]
    W = eigenvectors[:,ncols]
    return W
