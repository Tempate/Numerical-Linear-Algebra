import numpy as np
import scipy.linalg as sl


TOLERANCE = 1e-10


def qr_with_gram_schmidt(matrix):
    m, n = matrix.shape

    Q = np.eye(m)
    R = matrix

    for i in range(n):
        v = matrix[:, i]

        # The new unit-vector must be orthogonal to all 
        # previously calculated unit-vectors (qs)
        for j in range(i):
            q = Q[:, j]

            # Find the appropiate coefficient for orthogonality
            R[i,j] = np.dot(q, v)

            v -= R[i,j] * q

        # Save the new vector normalized
        Q[:, i] = v / sl.norm(v)

    return Q, R


def qr_with_reflections(matrix):
    m, n = matrix.shape

    Q = np.eye(m)
    R = matrix

    for i in range(n-1):
        dim = m - i

        # Find a reflection that transforms x into a 
        # direction parallel to an identity vector e
        x = R[i:, i]
        e = np.array([1] + [0] * (dim-1))

        u = x - sl.norm(x) * e
        v = u / sl.norm(u)

        # Reflection across the plane orthogonal to v
        P = np.eye(dim) - 2 * np.outer(v, v)

        # Add identity vectors for the dimensions that 
        # have already been computed
        P = np.pad(np.eye(i), (0, dim)) + np.pad(P, (i, 0))

        Q = Q @ P
        R = P @ R

    R[np.abs(R) < TOLERANCE] = 0

    return Q, R


def qr_with_rotations(matrix):
    m, n = matrix.shape

    Q = np.eye(m)
    R = matrix

    for i in range(n-1):
        for j in range(i+1, m):
            # Rotation to nullify the value at i,j
            G = givens_rotation(R, i, j, m)

            Q = Q @ G.T
            R = G @ R

    R[np.abs(R) < TOLERANCE] = 0

    return Q, R


def givens_rotation(matrix, i, j, n):
    a = matrix[i,i]
    b = matrix[j,i]

    r = np.sqrt(a**2 + b**2)

    cos = a/r
    sin = -b/r

    rotation = np.eye(n)

    rotation[i,i] = cos
    rotation[i,j] = -sin
    rotation[j,i] = sin
    rotation[j,j] = cos

    return rotation


def test_orthogonality(matrix):
    Qt_Q = np.dot(matrix.transpose(), matrix)

    # The norm of an orthogonal matrix must be 1
    assert np.isclose(np.linalg.norm(matrix, 2), 1)
    
    # If Q is orthogonal, then Q^t * Q must be close to the identity
    deviation = Qt_Q - np.identity(matrix.shape[1])
    assert np.isclose(np.linalg.norm(deviation, 2), 0)

    # The eigenvalues of an orthogonal matrix must be 1 or -1
    for eig in np.linalg.eigvals(Qt_Q):
        assert np.isclose(abs(eig), 1)

    # The determinant of an orthogonal matrix must be 1
    assert np.isclose(np.linalg.det(Qt_Q), 1)

    print("Successfully passed tests for size", matrix.shape)


def test_qr_decomposition(matrix, method):
    Q, R = method(matrix)

    assert np.allclose(matrix, np.matmul(Q, R))
    test_orthogonality(Q)
