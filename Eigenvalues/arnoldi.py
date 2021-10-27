import numpy.linalg as nl
import numpy as np
 

TOLERANCE = 1e-12


def arnoldi_iteration(A, b, n):
    # Approximate eigenvalues and eigenvectors of general 
    # matrices by constructing an orthonormal basis of the 
    # Krylov subspace (useful for sparse matrices).
    
    # Orthonormal basis of the Krylov subspace
    Q = np.zeros((A.shape[0],n+1))

    # A on basis Q (Hessenberg form)
    h = np.zeros((n+1,n))
    
    Q[:,0] = b / nl.norm(b)

    for k in range(n):
        # New candidate vector
        v = np.dot(A, Q[:,k])

        # Subtract the projections on previous vectors
        for j in range(k+1):
            h[j,k] = np.dot(Q[:,j].T, v)
            v = v - h[j,k] * Q[:,j]
        
        # Quit if the norm of v is zero
        if (h[k+1,k] := nl.norm(v)) < TOLERANCE:
            break
            
        Q[:,k+1] = v / h[k+1,k]

    return Q, h[:n,:]


def hessenberg(matrix):
    n = matrix.shape[0]
    b = np.random.rand(n)

    h = arnoldi_iteration(matrix, b, n)[1]
    h[np.abs(h) < TOLERANCE] = 0

    return h