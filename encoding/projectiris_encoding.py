import numpy as np

from math import sin, cos, pi, exp, pow

from utils.error_utils import SUCCESS, RESOLUTION_ERROR

#--------------------------------------------------------------------------------

GAUSSIAN_SCALE = 0.4770322291

SIN = 0
COS = 1

ENCODED_PIXELS = 1024                   # should be divisible by 32(int) or 64(long)

BITCODE_LENGTH = 2 * ENCODED_PIXELS     # each encoded pixel brings 2 bits to the bitcode

#--------------------------------------------------------------------------------


#ToDo: Python code here. Optimize it with cython or anything like that.
#ToDo: Optimize implementation. DO NOT COMPUTE what is not necessary (if noise => do not compute anything for that)
# returns the bitcode and its mask
def encode_iris(norm_img, mask_img, angular_resolution, radial_resolution):
    # getting image dimensions
    height, width = norm_img.shape

    #creating the bitcode and it's mask
    bit_code = np.zeros(BITCODE_LENGTH, np.uint8)
    bit_code_mask = np.zeros(BITCODE_LENGTH, np.uint8)

    # number of slices image is cut up into. Ideally angular slices should divide
    # 360, and size of bitCode without a remainder. More importantly, their product
    # should be divisible by 32
    angular_slices = angular_resolution
    radial_slices = ENCODED_PIXELS // angular_resolution

    # maximum filter size - set to 1/3 of image height to avoid large, uninformative
    # filters
    max_filter = height // 3

    # tracks the position which needs to be modified in the bitcode and bitcodemask
    bit_code_index = 0

    for r_slice in range(radial_slices):
        # works out which pixel in the image to apply the filter to
        # uniformly positions the centres of the filters between radius=3 and radius=height/2
        # does not consider putting a filter centre at less than radius=3, to avoid tiny filters
        radius = ((r_slice * (height - 6)) // (2 * radial_slices)) + 3

        # iet filter dimension to the largest filter that fits in the image
        filter_height = 2 * radius - 1 if radius < (height - radius) else 2 * (height - radius) - 1

        # if the filter size exceeds the width of the image then correct this
        if filter_height > width - 1:
            filter_height = width - 1

        # if the filter size exceeds the maximum size specified earlier then correct this
        if filter_height > max_filter:
            filter_height = max_filter

        # generating sinusoidal filters
        p_sine = generate_sinusoidal_filter(filter_height, SIN)
        p_cosine = generate_sinusoidal_filter(filter_height, COS)

        for a_slice in range(angular_slices):
            theta = a_slice

            bit_code[bit_code_index] = gabor_pixel(radius, theta, p_cosine, norm_img, mask_img)
            bit_code[bit_code_index + 1] = gabor_pixel(radius, theta, p_sine, norm_img, mask_img)

            # check whether the pixel itself is good
            if mask_img[radius, theta]:
                bit_code_mask[bit_code_index] = 1
            else:
                bit_code_mask[bit_code_index] = 0

            # check whether a filter is good or bad
            if not is_good_filter(radius, theta, filter_height, mask_img):
                bit_code_mask[bit_code_index] = 0

            # we're assuming that pairs of bits in the bitCodeMask are equal
            bit_code_mask[bit_code_index + 1] = bit_code_mask[bit_code_index]

            # incrementing the index
            bit_code_index += 2

    return SUCCESS, bit_code, bit_code_mask


def is_good_filter(radius, theta, filter_height, mask):
    good_ratio = 0.5  # ratio of good bits in a good filter

    height, width = mask.shape
    r_lb = max(0, radius - (filter_height // 2))
    r_ub = min(height, radius + (filter_height // 2) + 1)

    t_lb = max(0, theta - (filter_height // 2))
    t_ub = min(width, theta + (filter_height // 2) + 1)

    # check the mask of all pixels within the range of the filter
    ratio = np.average(mask[r_lb:r_ub, t_lb:t_ub])

    # if the ratio of good pixels to total pixels in the filter is good, return true
    return ratio >= good_ratio


#ToDo: Python code here. Optimize it with cython or anything like that.
#ToDo: The color assignment is non-pythonic code. Fix it.
def generate_heatmap(norm_img):
    if norm_img is None:
        return None

    # getting image dimensions
    radii, angles = norm_img.shape

    #creating heatmap image and mask
    heatmap = np.empty((radii, angles, 3), np.uint8)
    mask = np.ones((radii, angles), np.int32)   # all pixels are valid

    #generating filters
    filter_size = 9
    sin_filter = generate_sinusoidal_filter(filter_size, SIN)
    cos_filter = generate_sinusoidal_filter(filter_size, COS)

    for i in range(radii):
        for j in range(angles):
            if i <= (filter_size // 2) or i >= (radii - (filter_size // 2)):
                #setting black color
                heatmap.itemset(i, j, 0, 0)
                heatmap.itemset(i, j, 1, 0)
                heatmap.itemset(i, j, 2, 0)
            else:
                #applying filters
                real = gabor_pixel(i, j, cos_filter, norm_img, mask)
                imag = gabor_pixel(i, j, sin_filter, norm_img, mask)

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


#ToDo: Python code here. Optimize it with cython or anything like that.
def gabor_pixel(rho, phi, sinusoidal_filter, norm_img, mask_img):
    # size of the filter to be applied
    filter_size = sinusoidal_filter.shape[0]   # we assume that the filter is sqared

    # running total used for integration
    running_total = 0.0

    # translated co-ords within image (image_x, image_y)
    angles = norm_img.shape[1]

    for i in range(filter_size):
        for j in range(filter_size):
            # actual angular position within the image
            image_y = j + phi - (filter_size // 2)

            # allow filters to loop around the image in the angular direction
            image_y %= angles
            if image_y < 0:
                image_y += angles

            # actual radial position within the image
            image_x = i + rho - (filter_size // 2)

            # if the bit is good then apply the filter and add this to the sum
            if mask_img.item(image_x, image_y):
                running_total += sinusoidal_filter.item(i, j) * norm_img.item(image_x, image_y)

    # return true if +ve and false if -ve
    return 1 if running_total >= 0.0 else 0


#ToDo: Python code here. Optimize it with cython or anything like that.
def generate_sinusoidal_filter(size, sinusoidal_type):
    sum_row = 0.0
    sin_filter = np.empty((size, size), np.float64)

    if sinusoidal_type == SIN:
        wave_fun = lambda phi: sin(pi * phi / (size // 2))

    elif sinusoidal_type == COS:
        wave_fun = lambda phi: cos(pi * phi / (size // 2))

    # unknown filter type
    else:
        return None

    # filling first row
    for j in range(size):
        phi = j - (size // 2)
        wave_value = wave_fun(phi)
        sin_filter.itemset(0, j, wave_value)
        sum_row += wave_value

    # normalizing first row
    for j in range(size):
        old_value = sin_filter.item(0, j)
        sin_filter.itemset(0, j, old_value - (sum_row / size))

    #filling filter
    for i in range(1, size):
        for j in range(size):
            sin_filter.itemset(i, j, sin_filter.item(0, j))

    #generating gaussian filter
    gaussian_filter = generate_gaussian_filter(size)

    #multiplying both filters
    for i in range(size):
        for j in range(size):
            new_value = sin_filter.item(i, j) * gaussian_filter.item(i, j)
            sin_filter.itemset(i, j, new_value)

    # make every row have equal +ve and -ve
    for i in range(size):
        #computing row_sum
        row_sum = 0.0
        for j in range(size):
            row_sum += sin_filter.item(i, j)

        #normalizing
        for j in range(size):
            old_value = sin_filter.item(i, j)
            sin_filter.itemset(i, j, old_value - (row_sum / size))

    return sin_filter


#ToDo: Python code here. Optimize it with cython or anything like that.
def generate_gaussian_filter(size, peak=15.0):
    # Scale the constants so that gaussian is always in the same range
    # Uses alpha = dimension * (4sqrt(-ln(1/3)))**-1
    # The gaussian will have the value peak/3 at each of its edges
    # and peak/9 at its corners
    alpha = (size - 1) * GAUSSIAN_SCALE
    beta = alpha
    gaussian_filter = np.empty((size, size), np.float64)

    for i in range(size):
        rho = i - (size / 2)
        for j in range(size):
            phi = j - (size / 2)
            wave_value = peak * exp(-pow(rho, 2.0) / pow(alpha, 2.0)) * exp(-pow(phi, 2.0) / pow(beta, 2.0))
            gaussian_filter.itemset(i, j, wave_value)

    return gaussian_filter