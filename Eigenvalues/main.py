from qr_iteration import eigenvalues
from random_matrices import *

import numpy.linalg as nl


def test(matrix, tolerance=1e-8):
    real_eigs = nl.eig(matrix)[0]
    real_eigs.sort()

    calc_eigs = sorted(eigenvalues(matrix))

    assert np.allclose(real_eigs, calc_eigs, atol=tolerance)


for _ in range(10):
    test(random_symmetric_matrix(5))

print("[+] Passed tests for symmetric matrices")

test(random_symmetric_orthogonal_matrix(5))

print("[+] Passed tests for symmetric orthogonal matrices")