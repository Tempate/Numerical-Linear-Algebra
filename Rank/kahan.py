import numpy as np


def kahan_matrix(theta, size):
    SIN = np.sin(theta)
    COS = np.cos(theta)

    sin_matrix = np.diag([SIN**i for i in range(size)])
    cos_matrix = np.eye(size)

    for i in range(size):
        for j in range(i+1, size):
            cos_matrix[i,j] = -COS

    sr = sin_matrix @ cos_matrix

    return sr.T @ sr
