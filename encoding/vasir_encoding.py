import numpy as np

from math import exp, log, sqrt, sin, cos, ceil, pi

from utils.error_utils import SUCCESS

ENCODE_SCALES = 1           # number of filters to use in encoding
MIN_WAVE_LENGTH = 18        # base wavelength
MULT = 1                    # multiplicative factor between each filter (not applicable if using ENCODE_SCALES = 1)
SIGMA_ONF = 0.5             # bandwidth parameter


#ToDo: Python code here. Optimize it with cython or anything like that.
#ToDo: Optimize function. It has very very bad programming!!!
# Generates a biometric template from the normalised iris region, also generates
# corresponding noise mask
def encode_iris(polar_array, noise_array):
    n_scales = ENCODE_SCALES
    min_wave_length = MIN_WAVE_LENGTH
    mult = MULT
    sigma_onf = SIGMA_ONF

    #calling gabor convolve
    result = gabor_convolve(polar_array, n_scales, min_wave_length, mult, sigma_onf)
    E0, filter_sum, lenh, lenw = result

    polar_height, polar_width = polar_array.shape

    length = polar_width * 2 * n_scales

    template = np.zeros(length * polar_height, np.uint8)
    mask = np.zeros(length * polar_height, np.uint8)

    length2 = polar_width

    H1 = np.empty(lenw * lenh, np.uint8)
    H2 = np.empty(lenw * lenh, np.uint8)
    H3 = np.empty(lenw * lenh, np.uint8)

    for k in range(n_scales):
        E1 = E0[k]

        #ToDo: The actions performed at this for loops can be fused!!!

        # phase quantisation
        for i in range(lenh):
            for j in range(lenw):
                real_part = E1[i, j].real
                imag_part = E1[i, j].imag

                if real_part > 0:
                    H1[i * lenw + j] = 1
                else:
                    H1[i * lenw + j] = 0

                if imag_part > 0:
                    H2[i * lenw + j] = 1
                else:
                    H2[i * lenw + j] = 0

                # if amplitude is close to zero then phase data is not useful, so
                # mark off in the noise mask (0 => bad, 1 => good)
                if sqrt(imag_part * imag_part + real_part * real_part) < 0.0001:
                    H3[i * lenw + j] = 0
                else:
                    H3[i * lenw + j] = 1

        # building representation
        for i in range(length2):
            ja = 2 * n_scales * i
            for j in range(polar_height):
                index_1 = j * length + ja + 2 * k
                index_2 = index_1 + 1
                index_3 = j * polar_width + i

                # construct the biometric template
                template[index_1] = H1[index_3]
                template[index_2] = H2[index_3]

                # create noise mask
                value = noise_array[j, i] and H3[index_3]   # good if both are good
                mask[index_1] = value
                mask[index_2] = value

    # returning the template and the mask
    return SUCCESS, template, mask


def generate_heatmap(norm_img):
    if norm_img is None:
        return None

    # getting image dimensions
    radii, angles = norm_img.shape

    #creating heatmap image and mask
    heatmap = np.empty((radii, angles, 3), np.uint8)

    #getting codification
    result = gabor_convolve(norm_img, ENCODE_SCALES, MIN_WAVE_LENGTH, MULT, SIGMA_ONF)
    EO = result[0]
    heat_info = EO[0]   # infor for n_scales = 1

    for i in range(radii):
        for j in range(angles):
            if i <= 4 or i >= (radii - 4):
                # setting black color
                heatmap.itemset(i, j, 0, 0)
                heatmap.itemset(i, j, 1, 0)
                heatmap.itemset(i, j, 2, 0)
            else:
                # getting encoded pixel info
                real = 1 if heat_info[i, j].real >= 0.0 else 0
                imag = 1 if heat_info[i, j].imag >= 0.0 else 0

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
#ToDo: Optimize function. It has very very bad programming!!!
def gabor_convolve(im, n_scale, min_wave_length, mult, sigma_onf):

    # getting image dimensions
    rows, cols = im.shape

    #ToDo: Might be a bug here. Maybe it should be n_data % 2 == 1
    n_data = cols
    if n_data // 2 == 1:     # if there is an odd No. of data points
        n_data -= 1         # throw away the last one

    # creating radius array
    radius_count = n_data // 2 + 1
    radius = np.empty(radius_count, np.float64)

    # creating the log gabor array
    log_gabor = np.zeros(n_data, np.float64)

    # creating the filter sum
    filter_sum = np.zeros(n_data, np.float64)

    # creating the image fft
    image_fft = np.empty(n_data, np.complex)

    # creating the signal
    signal = np.empty(n_data, np.complex)

    # creating EO array
    EO = np.empty((n_scale, rows, n_data), np.complex)

    i = 0
    while i < radius_count:
        radius[i] = i / n_data // 2 / 2
        i += 1

    radius[0] = 1

    wave_length = min_wave_length   # initialize filter wavelength.

    # foreach scale.
    for s in range(n_scale):
        # construct the filter - first calculate the radial filter component.
        fo = 1.0 / wave_length      # centre frequency of filter.
        #rfo = fo / 0.5              # normalised radius from centre of frequency plane
        # corresponding to fo.

        for j in range(n_data // 2 + 1):
            log_fo = log(radius[j] / fo)
            log_sigma = log(sigma_onf)
            log_gabor[j] = exp(-log_fo * log_fo / (2 * log_sigma * log_sigma))

        log_gabor[0] = 0

        m_filter = log_gabor

        #ToDo: Optimize here. The previous for and this for can be done both in only one for
        for j in range(n_data // 2 + 1):
            filter_sum[j] += m_filter[j]

        # foreach row of the input image, do the convolution, back transform
        for r in range(rows):
            for j in range(n_data):
                signal[j] = complex(im[r, j], 0)

            # computing the fourier transform
            ft = fft(signal, n_data)

            # computing image fft
            for j in range(n_data):
                image_fft[j] = complex(ft[j].real * m_filter[j], ft[j].imag * m_filter[j])

            # save the ouput for each scale
            EO[s, r] = ifft(image_fft, n_data)

        # finally calculate Wavelength of next filter and process the next scale
        wave_length *= mult

    # applying fftshit
    fftshift(filter_sum, 2, (1, cols))

    EOh = rows
    EOw = n_data

    return EO, filter_sum, EOh, EOw


#ToDo: Python code here. Optimize it with cython or anything like that.
#ToDo: Optimize function. It has very very bad programming!!!
def fft(x, N):
    y = np.zeros(N, np.complex)

    # base case
    if N == 1:
        y[0] = x[0]
        return y

    # radix 2 Cooley-Tukey FFT
    if N % 2 != 0:
        dft(x, y, N)
        return y

    even = np.empty(N // 2, np.complex)
    odd = np.empty(N // 2, np.complex)

    #ToDo: Optimize, the following two for loops can be fused into one
    for k in range(N // 2):
        even[k] = x[2 * k]

    for k in range(N // 2):
        odd[k] = x[2 * k + 1]

    q = fft(even, N // 2)
    r = fft(odd, N // 2)

    for k in range(N // 2):
        kth = -2 * k * pi / N
        wk = complex(cos(kth), sin(kth))

        mul_real = wk.real * r[k].real - wk.imag * r[k].imag
        mul_imag = wk.real * r[k].imag + wk.imag * r[k].real
        mul = complex(mul_real, mul_imag)

        y[k] = q[k] + mul

        y[k + N // 2] = q[k] - mul

    return y


#ToDo: Python code here. Optimize it with cython or anything like that.
#ToDo: Optimize function. It has very very bad programming!!!
def ifft(x, N):
    # take conjugate
    for i in range(N):
        x[i] = x[i].conjugate()

    # compute forward FFT
    y = fft(x, N)

    #ToDo: The next two for loops can be fused into one
    # take conjugate again
    for i in range(N):
        y[i] = y[i].conjugate()

    # divide by N
    for i in range(N):
        y[i] = y[i] / N

    return y


#ToDo: Python code here. Optimize it with cython or anything like that.
#ToDo: Optimize function. It has very very bad programming!!!
def dft(x, y, N):
    # base case
    if N == 1:
        y[0] = x[0]

    for k in range(N):
        for j in range(N):
            kth = -2 * k * j * pi / N
            wk = complex(cos(kth), sin(kth))

            mul_real = wk.real * x[j].real - wk.imag * x[j].imag
            mul_imag = wk.real * x[j].imag + wk.imag * x[j].real
            mul = complex(mul_real, mul_imag)

            y[k] += mul


#ToDo: Python code here. Optimize it with cython or anything like that.
#ToDo: Optimize function. It has very very bad programming!!!
def fftshift(x, num_dims, size):
    count = 0

    y = x.copy()
    idx = np.empty(x.shape, np.int32)

    for k in range(num_dims - 1, num_dims):
        m = size[k]
        p = int(ceil(m / 2))

        for i in range(p + 1, m + 1):
            idx[count] = i - 1
            count += 1

        for i in range(1, p + 1):
            idx[count] = i - 1
            count += 1

    for i in range(count):
        x[i] = y[idx[i]]