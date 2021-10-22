from kahan import kahan_matrix
from rank import rank

A = kahan_matrix(1.2, 90)

print(rank(kahan_matrix(1.2, 90), "svd"))