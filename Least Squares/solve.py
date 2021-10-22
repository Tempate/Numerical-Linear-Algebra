import scipy.linalg as sl
import numpy as np


def solve_with_normal_equations(a, b):
    # Use cholesky to simplify the equations
    # A^t A x = A^t b   =>   R^t R x = A^t b
    r = sl.cholesky(a.T @ a)

    # Now the system is easy to solve since
    # R is an upper-triangular matrix
    return sl.solve(r, sl.solve(r.T, a.T @ b))


def solve_with_svd(a, b):
    u, s, v = sl.svd(a)
    sigma = sl.diagsvd(s, *a.shape)

    min_dim = min(a.shape)
    
    # Compute the right side with orthogonality
    # U E V^t x = b   =>   E V^t x = U^t b
    sols = (u.T @ b)[:min_dim, :min_dim]

    # Compute the left side's coefficients
    coeffs = (sigma @ v)[:min_dim, :min_dim]

    return sl.solve(coeffs, sols)


def solve_with_qr(a, b):
    q, r = sl.qr(a, mode="economic")

    # Compute the right side with orthogonality
    # And solve with backwards substitution
    return sl.solve(r, q.T @ b)