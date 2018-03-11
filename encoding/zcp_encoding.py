import cv2
import numpy as np

from encoding.fda_encoding import ComputeMfCircular

from utils.error_utils import SUCCESS, UNKNOWN_FAIL


def encode_iris(norm_img, order=16):
    # getting image dimensions
    height, width = norm_img.shape

    # computing Mf matrix
    A = ComputeMfCircular(width, height, order)

    # reshaping image (as a column vector)
    b = norm_img.reshape((height * width,))

    # changing datatype to float64
    b = b.astype(np.float64)

    # solving the corresponding SEL (Ax = b) by min squares
    result, x = cv2.solve(src1=A, src2=b, flags=cv2.DECOMP_QR)

    # if there was an error of some kind
    if not result:
        return UNKNOWN_FAIL, None, None

    # returning the solution of the SEL if any
    return SUCCESS, x, None
