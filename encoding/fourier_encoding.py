import numpy as np
import numpy.fft as fmath

from utils.error_utils import SUCCESS


def encode_iris(norm_img, mask_img, angular_resolution, radial_resolution):
    # getting image dimensions
    height, width = norm_img.shape

    # getting radii and angles
    radii, angles = height, width

    #creating the bitcode and it's mask
    bit_code = np.empty(2 * width * height, np.uint8)
    bit_code_mask = np.empty(2 * width * height, np.uint8)

    # ecoding image
    encoded_img = fourier_image(norm_img)

    # creating index
    index = 0
    for radius in range(radii):
        for theta in range(angles):
            real = 1 if encoded_img[radius, theta].real >= 0.0 else 0
            imag = 1 if encoded_img[radius, theta].imag >= 0.0 else 1

            bit_code[index] = real
            bit_code[index + 1] = imag

            if mask_img[radius, theta]:
                bit_code_mask[index] = 1
                bit_code_mask[index + 1] = 1
            else:
                bit_code_mask[index] = 0
                bit_code_mask[index + 1] = 0

            index += 2

    return SUCCESS, bit_code, bit_code_mask


def fourier_image(img):
    return fmath.fft2(img)


def generate_heatmap(norm_img):
    if norm_img is None:
        return None

    # getting image dimensions
    radii, angles = norm_img.shape

    #creating heatmap image and mask
    heatmap = np.empty((radii, angles, 3), np.uint8)

    # ecoding image
    encoded_img = fourier_image(norm_img)

    for i in range(radii):
        for j in range(angles):
            if i <= 4 or i >= (radii - 4):
                #setting black color
                heatmap.itemset(i, j, 0, 0)
                heatmap.itemset(i, j, 1, 0)
                heatmap.itemset(i, j, 2, 0)
            else:
                real = 1 if encoded_img[i, j].real >= 0.0 else 0
                imag = 1 if encoded_img[i, j].imag >= 0.0 else 1

                if imag:
                    if real:
                        heatmap.itemset(i, j, 0, 91)
                        heatmap.itemset(i, j, 1, 102)
                        heatmap.itemset(i, j, 2, 166)
                    else:
                        heatmap.itemset(i, j, 0, 91)
                        heatmap.itemset(i, j, 1, 140)
                        heatmap.itemset(i, j, 2, 77)
                else:
                    if real:
                        heatmap.itemset(i, j, 0, 120)
                        heatmap.itemset(i, j, 1, 120)
                        heatmap.itemset(i, j, 2, 120)
                    else:
                        heatmap.itemset(i, j, 0, 217)
                        heatmap.itemset(i, j, 1, 179)
                        heatmap.itemset(i, j, 2, 145)

    return heatmap