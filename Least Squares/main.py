import numpy as np

from solve import *


with open("signal.dat") as f:
    points = f.read().splitlines()
    points = [point.split(", ") for point in points]


def gen_matrices(points):
    xs = []
    ys = []

    for x, y in points:
        x, y = float(x), float(y)

        sin  = np.sin(x)
        cos  = np.cos(x)
        sin2 = np.sin(2 * x)
        cos2 = np.cos(2 * x)
        
        xs.append([sin, cos, sin2, cos2])
        ys.append([y])

    return np.array(xs), np.array(ys)


x, y = gen_matrices(points)

print(solve_with_qr(x, y))