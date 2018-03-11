import cv2
import numpy as np

from encoding.fda_encoding import ComputeMfAnnular

from utils.error_utils import SUCCESS, UNKNOWN_FAIL


def encode_iris(norm_img, mask_img, order=16, eps_lb=0.25, eps_ub=1.0):
    # getting image dimensions
    height, width = norm_img.shape

    # computing Mf matrix
    A = ComputeMfAnnular(width, height, order, eps_lb, eps_ub)

    # reshaping image (as a column vector)
    b = norm_img.reshape((height * width,))

    # changing datatype to float64
    b = b.astype(np.float64)

    # # applying mask to the Mf matrix and the normalized image
    # mask_img = mask_img.reshape((height * width,))
    # # creating the filter
    # mask = mask_img[:] == 1
    #
    # # filtering Mf and normalized image
    # A = A[mask]     # getting only "good" rows
    # b = b[mask]     # getting only "good" rows

    # solving the corresponding SEL (Ax = b) by min squares
    result, x = cv2.solve(src1=A, src2=b, flags=cv2.DECOMP_QR)

    # if there was an error of some kind
    if not result:
        return UNKNOWN_FAIL, None, None

    # returning the solution of the SEL if any
    return SUCCESS, x, None
