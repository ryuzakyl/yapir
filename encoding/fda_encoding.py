import os
from math import pi

import numpy as np

from fda.zernike_annular_polynomial import ZernikeAnnularSingle as ZernikeAnnular
from fda.zernike_circular_polynomial import ZernikeCircularSingle as ZernikeCircular

# ----------------------------------------------------------------------------------

# extension of numpy arrays
ext = "npy"

# mfs matrices path
mfs_path = "./encoding/mfs"

# ----------------------------------------------------------------------------------


def ComputeMfCircular(width, height, order):
    # creating mf matrix name
    mf_name = get_mf_circ_name(width, height, order)
    mf_path = "%s/%s.%s" % (mfs_path, mf_name, ext)

    # if already computed, then return it
    if exists_mf(mf_path):
        return np.load(mf_path)

    # computing size
    size = width * height

    #creating mf matrix (size x order)
    Mf = np.empty((size, order), np.float64)

    ub = order + 1
    for j in range(1, ub):
        Mv = FillRectCircular(width, height, order)
        Vv = Mv.reshape((size,))
        Mf[:, j - 1] = Vv

    # saving the computed Mf matrix
    np.save(mf_path, Mf)

    #returning the Mf matrix
    return Mf


def ComputeMfAnnular(width, height, order, eps_lb, eps_ub):
    # creating mf matrix name
    mf_name = get_mf_annu_name(width, height, order, eps_lb, eps_ub)
    mf_path = "%s/%s.%s" % (mfs_path, mf_name, ext)

    # if already computed, then return it
    if exists_mf(mf_path):
        return np.load(mf_path)

    # computing size
    size = width * height

    #creating mf matrix (size x order)
    Mf = np.empty((size, order), np.float64)

    ub = order + 1
    for j in range(1, ub):
        Mv = FillRectAnnular(width, height, j, eps_lb, eps_ub)
        Vv = Mv.reshape((size,))
        Mf[:, j - 1] = Vv

    # saving the computed Mf matrix
    np.save(mf_path, Mf)

    #returning the Mf matrix
    return Mf


def FillRectCircular(width, height, order):
    # computing iris ring radius
    i_radius = 1.0  # unitary disk

    # computing deltha of radius (amount to add to the radius)
    dr = i_radius // height

    # creating rectangle
    rect = np.empty((height, width), np.float64)

    for a in range(width):
        # computing theta
        theta = a * (2 * pi) / width

        for r in range(height):
            # computing rho
            rho = dr + r * dr

            # # saving value of zernike circular polynomial in rect
            rect[r, a] = ZernikeCircular(order, rho, theta)

    return rect


def FillRectAnnular(width, height, order, eps_lb, eps_ub):
    # computing iris ring radius
    i_radius = eps_ub - eps_lb

    # computing deltha of radius (amount to add to the radius)
    dr = i_radius // (height - 1)

    # creating rectangle
    rect = np.empty((height, width), np.float64)

    for a in range(width):
        # computing theta
        theta = a * (2 * pi) / width
        for r in range(height):
            # computing rho
            rho = eps_lb + r * dr

            # saving value of zernike annular polynomial in rect
            rect[r, a] = ZernikeAnnular(order, rho, theta, eps_lb, eps_ub)

    # returning the evaluated rectangle
    return rect


def exists_mf(mf_path):
    return os.access(mf_path, os.F_OK)


def get_mf_circ_name(width, height, order):
    return "%i_%i_%i" % (width, height, order)


def get_mf_annu_name(width, height, order, eps_lb, eps_ub):
    return "%i_%i_%i_%.2f_%.2f" % (width, height, order, eps_lb, eps_ub)