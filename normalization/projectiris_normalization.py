import cv2
import numpy as np

from math import pi, sin, cos
from utils.math_utils import is_between_parabolas_coords
from utils.image_utils import valid_pixel
from utils.error_utils import SUCCESS
#--------------------------------------------------------------------------------

ANGULAR_RESOLUTION = 256    # default angular resolution
RADIAL_RESOLUTION = 64      # default radial resolution

THRESHOLD_SPECULAR = 220
THRESHOLD_BLACK_BIT = 40    # this is usually the pupil threshold

#--------------------------------------------------------------------------------


#Assumes that the pupil and iris are concentric, and uses only the pupil center
def normalize_iris(img, angular_resolution, radial_resolution, pupil_center, pupil_radius, iris_center, iris_radius, upper_eyelid, lower_eyelid):
    angles = angular_resolution                 # amount of angles to map (<= 360)
    radii = radial_resolution                   # amount of radius samples
    r_annular = int(iris_radius - pupil_radius)  # iris width (the width of the iris ring)

    img_height = img.shape[0]   # getting the image height
    xp, yp = pupil_center       # (xp, yp) is the pupil center

    # creating the normalized image
    norm_image = np.empty((radii, angles), np.uint8)
    mask_image = np.empty((radii, angles), np.uint8)     # 1 if valid pixel, 0 otherwise

    #ToDo: Python code here. Optimize it with cython or anything like that.
    #foreach angle
    for col in range(angles):
        #computing the current angle
        theta = col * (2 * pi) / angles     # simple "three rule" (for the cubans)
        cos_theta = cos(theta)
        sin_theta = sin(theta)

        #foreach radius
        for row in range(radii):
            #getting pixel location in the original image
            r = pupil_radius + r_annular * row / (radii - 1)
            x = xp + r * cos_theta
            y = yp - r * sin_theta  # -r*sin(theta) => anticlockwise radial selection

            #getting the pixel value
            pixel_value = img.item(y, x)    # indexed first by rows, then by columns

            # If pixel out of bounds
            if not valid_pixel(img, x, y):
                mask_image.itemset(row, col, 0)

            # If pixel is a black bit
            elif pixel_value < THRESHOLD_BLACK_BIT:
                mask_image.itemset(row, col, 0)

            # If pixel is a specular reflection
            elif pixel_value > THRESHOLD_SPECULAR:
                mask_image.itemset(row, col, 0)

            # If pixel doesn't belong inside the two parabolas
            elif not is_between_parabolas_coords(upper_eyelid, lower_eyelid, x, img_height - y):
                mask_image.itemset(row, col, 0)

            # Everything is OK
            else:
                mask_image.itemset(row, col, 1)

            #setting the pixel in the normalized iris image
            norm_image.itemset(row, col, pixel_value)

    #returning the unwrapped image and the corresponding mask
    return SUCCESS, norm_image , mask_image