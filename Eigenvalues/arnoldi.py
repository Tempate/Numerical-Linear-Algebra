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

    for k in range(1,n+1):
        # New candidate vector
        v = np.dot(A, Q[:,k-1])

        # Subtract the projections on previous vectors
        for j in range(k):
            h[j,k-1] = np.dot(Q[:,j].T, v)
            v = v - h[j,k-1] * Q[:,j]
        
        h[k,k-1] = nl.norm(v)
        
        # Quit if the norm of v is zero
        if h[k,k-1] <= TOLERANCE:
            break
            
        Q[:,k] = v / h[k,k-1]

    return Q, h[:n,:]


def hessenberg(matrix):
    n = matrix.shape[0]
    b = np.random.rand(n)

    h = arnoldi_iteration(matrix, b, n)[1]
    h[np.abs(h) < TOLERANCE] = 0

    return h