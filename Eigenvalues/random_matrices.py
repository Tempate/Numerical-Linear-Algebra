import numpy.linalg as nl
import numpy as np


def random_matrix(size):
    return np.random.rand(size, size)


def random_symmetric_matrix(size):
    matrix = random_matrix(size)
    return (matrix + matrix.T) / 2


def random_orthogonal_matrix(size):
    q,r = nl.qr(random_matrix(size))
    return q @ np.diag(np.sign(np.diag(r)))


def random_symmetric_orthogonal_matrix(size):
    orth = random_orthogonal_matrix(size)
    diag = [np.random.choice([-1, 1]) for _ in range(size)]
    return orth @ (diag * np.eye(size)) @ orth.T