from math import ceil, sqrt, atan2
import numpy as np
from zernike_annular_polynomial import ZernikeAnnularDouble as ZernikeAnnular


def FillGridSingle(j, diameter, eps_lb, eps_ub):
    n = int(ceil((-3 + sqrt(9 + 8 * j)) / 2))
    m = 2 * j - n * (n + 2)

    return FillGridDouble(n, m, diameter, eps_lb, eps_ub)


def FillGridDouble(n, m, diameter, eps_lb, eps_ub):
    # checking that n >= |m|
    if abs(m) > n:
        return np.empty(0)

    # computing radius
    radius = diameter / 2.0

    # creating the grid
    grid = np.empty((diameter, diameter), np.float64)

    for i in range(diameter):
        for j in range(diameter):
            X = (j - radius) / radius   # normalizing grid coordinates in [-1; 1]
            Y = (radius - i) / radius   # normalizing grid coordinates in [-1; 1]

            # computing rho
            rho = sqrt(pow(X, 2) + pow(Y, 2))

            # computing theta
            theta = atan2(Y, X)

            grid[i, j] = ZernikeAnnular(n, m, rho, theta, eps_lb, eps_ub)

    # returning the computed grid
    return grid


def norm_img(img, diameter, eps_lb, eps_ub):
    height, width = img.shape

    min_val = img.min()
    min_val = abs(min_val)
    img += min_val

    max_val = img.max()
    img = img / max_val
    img *= 255

    radius = diameter / 2.0

    result = np.empty((height, width), np.uint8)

    for i in range(height):
        for j in range(width):
            X = (j - radius) / radius   # normalizing grid coordinates in [-1; 1]
            Y = (radius - i) / radius   # normalizing grid coordinates in [-1; 1]

            # computing rho
            rho = sqrt(pow(X, 2) + pow(Y, 2))

            if rho < eps_lb or rho > eps_ub:
                result[i, j] = 200
            else:
                result[i, j] = int(img[i, j])

    return result
