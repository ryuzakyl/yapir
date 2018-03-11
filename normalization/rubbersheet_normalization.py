import cv2
import numpy as np

from math import pi, sin, cos, atan, sqrt
from utils.math_utils import is_between_parabolas_coords
from utils.image_utils import valid_pixel
from utils.error_utils import SUCCESS

#--------------------------------------------------------------------------------

ANGULAR_RESOLUTION = 256    # default angular resolution
RADIAL_RESOLUTION = 64      # default radial resolution

THRESHOLD_SPECULAR = 220
THRESHOLD_BLACK_BIT = 40    # this is usually the pupil threshold

#--------------------------------------------------------------------------------


# The pupil and iris might not be concentric
def normalize_iris(img, angular_resolution, radial_resolution, pupil_center, pupil_radius, iris_center, iris_radius, upper_eyelid, lower_eyelid):
    angles = angular_resolution          # amount of angles to map (<= 360)
    radii = radial_resolution            # iris width (the width of the iris ring)
    radii2 = radii + 2                   # radial resolution plus 2

    img_height = img.shape[0]   # getting the image height
    xp, yp = pupil_center       # (xp, yp) is the pupil center
    xi, yi = iris_center        # (xi, yi) is the iris center
    rp = pupil_radius           # rp is the radius of the pupil
    ri = iris_radius            # ri is the radius of the iris

    # creating the normalized image
    norm_image = np.empty((radii, angles), np.uint8)
    mask_image = np.empty((radii, angles), np.uint8)     # 1 if valid pixel, 0 otherwise

    # computing centers offset
    ox = xp - xi    # offstet of pupil and iris centers in the x axis
    oy = yp - yi    # offstet of pupil and iris centers in the y axis

    if ox < 0:
        sgn = -1
        phi = atan(oy / ox)
    elif ox > 0:
        sgn = 1
        phi = atan(oy / ox)
    else:
        sgn = 1 if oy > 0 else -1
        phi = pi / 2.0

    #computing alpha
    alpha = ox * ox + oy * oy

    #ToDo: Python code here. Optimize it with cython or anything like that.
    #foreach angle
    for col in range(angles):
        # computing the current angle
        theta = col * (2 * pi) / (angles - 1)       # simple "three rule" (for the cubans)
        cos_theta = cos(theta)
        sin_theta = sin(theta)

        # computing beta
        beta = sgn * cos(pi - phi - theta)

        # computing the radius of the iris ring for angle theta (see Libor Masek's thesis)
        r_prime = sqrt(alpha) * beta + sqrt(alpha * beta * beta - (alpha - ri * ri))
        r_prime -= rp

        #foreach radius
        for row in range(radii2):
            # computing radius from pupil center to the current sampled point
            r = rp + r_prime * row / (radii2 - 1)

            #excluding the first and last rows (pupil/iris border and iris/sclera border)
            if 0 < row < radii2 - 1:
                #getting pixel location in the original image
                x = int(xp + r * cos_theta)
                y = int(yp - r * sin_theta)

                #getting the pixel value
                pixel_value = img.item(y, x)    # indexed first by rows, then by columns

                # If pixel out of bounds
                if not valid_pixel(img, x, y):
                    mask_image.itemset(row - 1, col, 0)

                # If pixel is a black bit
                elif pixel_value < THRESHOLD_BLACK_BIT:
                    mask_image.itemset(row - 1, col, 0)

                # If pixel is a specular reflection
                elif pixel_value > THRESHOLD_SPECULAR:
                    mask_image.itemset(row - 1, col, 0)

                # If pixel doesn't belong inside the two parabolas
                elif not is_between_parabolas_coords(upper_eyelid, lower_eyelid, x, img_height - y):
                    mask_image.itemset(row - 1, col, 0)

                # Everything is OK
                else:
                    mask_image.itemset(row - 1, col, 1)

                #setting the pixel in the normalized iris image
                norm_image.itemset(row - 1, col, pixel_value)

    #returning the unwrapped image and the corresponding mask
    return SUCCESS, norm_image , mask_image