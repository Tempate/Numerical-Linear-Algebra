import scipy.linalg as sl
import numpy.linalg as nl
import numpy as np


def eigenvalues(matrix, tolerance=1e-8):
    if matrix.shape[0] <= 1:
        return np.diag(matrix)

    # Make the original matrix tridiagonal
    new_matrix = sl.hessenberg(matrix)
    I = np.eye(matrix.shape[0])

    for k in range(1,6):
        # Approximate an eigenvalue
        mu = new_matrix[-1,-1]

        # Rayleigh shift
        shift = mu * I

        q,r = sl.qr(new_matrix - shift)
        new_matrix = r @ q + shift

        for i in range(1, new_matrix.shape[0]):
            # Break the matrix into submatrices when an 
            # approximation is close to the eigenvalue
            if abs(new_matrix[i-1,i]) < tolerance:
                eigs1 = eigenvalues(new_matrix[:i,:i])
                eigs2 = eigenvalues(new_matrix[i:,i:])
                return np.concatenate((eigs1, eigs2))

    return np.diag(new_matrix)
