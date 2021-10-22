import scipy.linalg as sl
import numpy as np


matrix = sl.hilbert(50)

u, s, v = sl.svd(matrix)

# We want to minimize ‖db‖ / ‖b‖. To do it, we choose b 
# to be the first left-singular vector, the one that 
# undergoes the most stretching, and db to be the last 
# left-singular vector, the one that undergoes the least
# stretching.
b  = u[:, 0]
db = u[:, -1]

x  = sl.solve(matrix, b)
dx = sl.solve(matrix, b + db) - x

f1 = sl.norm(dx) / sl.norm(x)
f2 = sl.norm(db) / sl.norm(b)

print(np.linalg.cond(matrix) - f1/f2)