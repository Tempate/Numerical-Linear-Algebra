import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.linalg as nl
import numpy as np


matrix = np.array([
    [5, 0, 0, -1],
    [1, 0, -1, 1],
    [-1.5, 1, -2, 1],
    [-1, 1, 3, -3]
], dtype=float)


def scale(matrix, factor):
    new_matrix = np.zeros(matrix.shape)

    for j in range(matrix.shape[0]):
        for i in range(matrix.shape[1]):
            new_matrix[i,j] = matrix[i,j]

            if i != j:
               new_matrix[i,j] *= factor

    return new_matrix


def visualize(matrix):
    real, imag = [], []
    radiuses = []

    eigenvalues = nl.eig(matrix)[0]
    diagonals = np.diag(matrix)
    discs = eigendiscs(matrix)

    for i in range(len(diagonals)):
        # Add diagonal point
        real.append(diagonals[i].real)
        imag.append(diagonals[i].imag)
        radiuses.append(50)

        # Add Gerschgorin's disc
        real.append(diagonals[i].real)
        imag.append(diagonals[i].imag)
        radiuses.append(1500 * discs[i])

        # Add eigenvalue
        real.append(eigenvalues[i].real)
        imag.append(eigenvalues[i].imag)
        radiuses.append(50)

    faces = ['blue', 'none', 'red'] * len(diagonals)
    edges = ['blue', 'blue', 'red'] * len(diagonals)

    return plt.scatter(real, imag, s=radiuses, facecolors=faces, edgecolors=edges, animated=True)


def eigendiscs(matrix):
    radius = lambda j: sumabs(j) - abs(matrix[j,j])
    sumabs = lambda j: sum(abs(e) for e in matrix[j,:])

    return [radius(j) for j in range(matrix.shape[0])]


if __name__ == "__main__":
    figure = plt.figure()
    frames = []

    dx = 600

    for factor in range(dx+1): 
        plot = visualize(scale(matrix, factor / dx))
        frames.append([plot])

    ani = animation.ArtistAnimation(figure, frames, interval=99, blit=True, repeat_delay=1000)
    ani.save('movie.mp4')
    plt.show()