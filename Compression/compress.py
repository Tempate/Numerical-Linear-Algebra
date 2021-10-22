from matplotlib.image import imread
from matplotlib import pyplot

import scipy.linalg as sl
import numpy as np


image = imread("./kvinna.jpg")


def show_image(image):
    pyplot.imshow(image)
    pyplot.show()


def compress(image):
    u, s, v = sl.svd(image)
    sigma = sl.diagsvd(s, *image.shape)

    # Ignore small values
    sigma[abs(sigma) < 1e-10] = 0.0

    return u @ sigma @ v


show_image(compress(image))
