import cv2
import numpy as np
from math import sqrt, atan2, pi, pow, ceil, floor, cos, sin

from segmentation.projectiris_segmentation import find_eyelids

from utils.error_utils import SUCCESS, WRONG_IMAGE_FORMAT, PUPIL_DETECTION_FAILED, IRIS_DETECTION_FAILED, EYELIDS_DETECTION_FAILED

#--------------------------------------------------------------------------------

MAX_PUPIL_RADIUS = 75   # maximum radius of pupil   (need to be changed from db to db)
MAX_IRIS_RADIUS = 150   # maximum iris radius   (need to be changed from db to db)

RATIO_R = 13

STD_THRESHOLD = 47

WHITE = 255
BLACK = 0

FRAC_OFF_LENGTH = 181
ADJ_PRECISION = 0.0000000000005


# NOTE: I assume that nScales = 1 because it only changes from 1 iif the width of the picture is bigger than 600x400

#--------------------------------------------------------------------------------


def segment_iris(eye_img):
    #image must be in grayscale
    if eye_img.dtype != np.uint8:
        return WRONG_IMAGE_FORMAT, None

    # ----------------------------------------------------------------------------

    #trying to detect pupil
    pupil_data = find_pupil(eye_img, MAX_PUPIL_RADIUS)
    if pupil_data is None:
        return PUPIL_DETECTION_FAILED, None

    # ----------------------------------------------------------------------------

    # getting pupil data
    pupil_center, r_pupil = pupil_data
    x_pupil, y_pupil = pupil_center

    #detecting iris
    top_left, bottom_right = get_iris_roi_rect(eye_img, x_pupil, y_pupil, r_pupil, MAX_IRIS_RADIUS)

    x_top, y_top = top_left
    x_bottom, y_bottom = bottom_right

    # the rectangle must be valid
    if x_top - x_bottom == 0 or y_top - y_bottom == 0:
        return IRIS_DETECTION_FAILED, (pupil_data, None, None)

    # getting the iris roi
    iris_roi = eye_img[y_top:y_bottom, x_top:x_bottom]

    # define maximum distance between pupil and iris center position
    center_adjust = MAX_IRIS_RADIUS // 4

    # find iris circle using Hough Transform
    iris_data = find_iris(iris_roi, r_pupil, MAX_IRIS_RADIUS)
    if iris_data is None:
        return IRIS_DETECTION_FAILED, (pupil_data, None, None)

    # getting iris data
    iris_center, r_iris = iris_data
    x_iris, y_iris = iris_center

    # getting coordinates in original image
    x_iris, y_iris = get_origin_points(x_top, y_top, x_iris, y_iris)

    # setting new values
    iris_center = x_iris, y_iris
    iris_data = iris_center, r_iris

    # ----------------------------------------------------------------------------

    # detecting eyelids [NOTE: i'm gonna use the eyelid detection of project iris]
    eyelids_data = find_eyelids(eye_img, (x_pupil, y_pupil))
    if eyelids_data is None:
        return EYELIDS_DETECTION_FAILED, (pupil_data, iris_data, None)

    # ----------------------------------------------------------------------------

    #all went well
    return SUCCESS, (pupil_data, iris_data, eyelids_data)


def find_pupil(img, limit_radius):
    # make a copy of the given image
    gray_img = img.copy()

    # setup the parameters to avoid noise caused by reflections and eyelashes covering the pupil
    size = 0.0

    # initialize for Closing and Opening process
    close_itr = 0  # dilate->erode
    open_itr = 0  # erode->dilate

    # for classical still eye images
    size = float(RATIO_R - 1)
    close_itr = 2
    open_itr = 3

    # find the minimum intensity within the image - used to determine the threshold
    min_value, _, _, _ = cv2.minMaxLoc(gray_img)

    # future work => consider to remove the reflections here

    # get the threshold for detecting the pupil
    threshold = get_threshold(gray_img, int(min_value))

    # initialize values
    center_x = 0
    center_y = 0
    radius = 0

    # originally, we used 1 for m value.
    # if you increase m value, it would be faster, however, the results would be slightly worst.
    m = 2
    while size > 2:
        i = threshold
        while i >= 0:
            center_x, center_y, radius = get_coordinates(gray_img, close_itr, open_itr, i, limit_radius, size)

            if center_x > 0 and center_y > 0 and radius > 0:
                return (center_x, center_y), radius

            i -= m

        size -= 1
        close_itr += 1

    # second attempt
    size = RATIO_R - 1
    close_itr = 0
    while size > 1:
        open_itr += 3  # count region

        for i in range(threshold + 5, threshold + 30):
            center_x, center_y, radius = get_coordinates(gray_img, close_itr, open_itr, i, limit_radius, size)

            if center_x > 0 and center_y > 0 and radius > 0:
                return (center_x, center_y), radius

        size -= 1

    return None


def find_iris(img, pupil_radius, u_iris_radius):
    # scaling factor to speed up the Hough transform
    scaling = 0.4

    # use the pupil's radius as a minimum for the potential Iris' circle radius
    ratio_size = pupil_radius / u_iris_radius
    l_iris_radius = int(pupil_radius + u_iris_radius / (ratio_size * 12))   # minimum radius of a potential iris
    if ratio_size >= 0.22:
        i = 0.22
        while i < 0.5:
            if ratio_size < i:
                l_iris_radius = int(pupil_radius + u_iris_radius / (i * 10))
                break

            i += 0.02

    #storing at most the u_iris_radius
    l_iris_radius = min(l_iris_radius, u_iris_radius)

    low_thres = 0.18
    hi_thres = 0.19

    # find the iris boundaries
    center_x, center_y, radius = find_circle(img, l_iris_radius, u_iris_radius, scaling, 2.0, hi_thres, low_thres, 1.00, 0.00)

    # there was a problem finding the iris circle
    if center_x == 0 or center_y == 0 or radius == 0:
        return None

    #assuming it all went well
    return (center_x, center_y), radius


def get_threshold(img, min_value):
    _, sigma = cv2.meanStdDev(img)
    std = sigma[0, 0]

    if std < STD_THRESHOLD:
        return min_value + 25   # usually poor quality->orgin 25
    else:
        return min_value + 35   # usually good quality->orgin 35


def get_coordinates(gray_img, close_itr, open_itr, threshold, limit_radius, size):
    dest_img = gray_img.copy()

    # thresholding the image
    cv2.threshold(src=dest_img, thresh=threshold, maxval=WHITE, type=cv2.THRESH_BINARY, dst=dest_img)

    # smoothing the image with a gaussian filter
    dest_img = cv2.GaussianBlur(dest_img, ksize=(5, 5), sigmaX=0)

    # start the morphological operators
    m_pSE = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3, 3), anchor=(1, 1))
    cv2.morphologyEx(src=dest_img, op=cv2.MORPH_CLOSE, kernel=m_pSE, dst=dest_img, iterations=close_itr)
    cv2.morphologyEx(src=dest_img, op=cv2.MORPH_OPEN, kernel=m_pSE, dst=dest_img, iterations=open_itr)

    # finding contours
    contours, hierarchy = cv2.findContours(dest_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_TC89_L1)

    # set up the min. contours to ignore noise within the binary image
    min_count = 16  # 16 * n_scales, but n_scales = 1

    # // Find the first and second maximum contour
    max_count = get_max_count(contours)

    # get the pupil center and radius
    pupil_data = get_pupil_position(contours, min_count, max_count, limit_radius, size)

    return pupil_data


def get_pupil_position(contours, min_count, max_count, limit_radius, size):
    center_x = 0
    center_y = 0
    radius = 0

    for item in contours:
        count = len(item)

        if count <= min_count or count != max_count:
            continue

        box = cv2.fitEllipse(item)

        x, y = box[0]
        center_x = int(x)
        center_y = int(y)

        h, w = box[1]
        height = int(h)
        width = int(w)

        # we assume that the pupil is the perfect circle
        radius = get_radius(width, height, limit_radius, size)

        # use below if the radius is bigger than the limitRadius
        if 0 < radius < limit_radius:
            break

    return center_x, center_y, radius


def get_max_count(contours):
    contours.sort(descending)

    count = len(contours)
    return len(contours[0]) if count >= 1 else None


def get_radius(width, height, limit_radius, size):
    if height > width:
        long_radius = height
        short_radius = width
    else:
        long_radius = width
        short_radius = height

    if not short_radius < (long_radius // RATIO_R) * size:
        radius = int(round(long_radius / 2) + 1)
    else:
        radius = limit_radius + 1

    return radius


def descending(x, y):
    return len(y) - len(x)


def get_iris_roi_rect(img, x_pupil, y_pupil, r_pupil, radius_iris_max):
    x_top = 0
    y_top = 0

    x_bottom = 0
    y_bottom = 0

    img_height, img_width = img.shape

    rect_width = max(r_pupil, radius_iris_max)
    rect_height = max(r_pupil, radius_iris_max)

    if x_pupil > 0 and y_pupil > 0 or r_pupil and 0:
        x_top = max(0, x_pupil - rect_width)
        y_top = max(0, y_pupil - rect_height)

        x_bottom = min(img_width, x_pupil + rect_width)
        y_bottom = min(img_height, y_pupil + rect_height)

    return (x_top, y_top), (x_bottom, y_bottom)


def get_origin_points(x_top, y_top, x_iris, y_iris):
    return x_top + x_iris, y_top + y_iris

# /*% findcircle - returns the coordinates of a circle in an image using the Hough transform
# % and Canny edge detection to create the edge map.
# %
# % Usage:
# % [row, col, r] = findcircle(image,lradius,uradius,scaling, sigma, hithres, lowthres, vert, horz)
# %
# % Arguments:
# %	image		    - the image in which to find circles
# %	lradius		    - lower radius to search for
# %	uradius		    - upper radius to search for
# %	scaling		    - scaling factor for speeding up the
# %			          Hough transform
# %	sigma		    - amount of Gaussian smoothing to
# %			          apply for creating edge map.
# %	hithres		    - threshold for creating edge map
# %	lowthres	    - threshold for connected edges
# %	vert		    - vertical edge contribution (0-1)
# %	horz		    - horizontal edge contribution (0-1)
# %
# % Output:
# %	circleiris	    - centre coordinates and radius
# %			          of the detected iris boundary
# %	circlepupil	    - centre coordinates and radius
# %			          of the detected pupil boundary
# %	imagewithnoise	- original eye image, but with
# %			          location of noise marked with
# %			          NaN values
# %


#ToDo: Change the calls to canny, adjust_gamma, non_max_suppression, and hys_thresh for a call to OpenCV canny
def find_circle(image, l_radius, u_radius, scaling, sigma, hi_thres, low_thres, vert, horz):
    l_rad_sc = round_nd(l_radius * scaling)
    u_rad_sc = round_nd(u_radius * scaling)
    rd = round_nd(u_radius * scaling - l_radius * scaling)

    # generate the edge image
    gradient, or_nd = canny(image, sigma, scaling, vert, horz)

    # adjusting gamma
    I3 = adjust_gamma(gradient, 1.9)

    # performing non-maxima suppression
    I4 = non_max_suppression(I3, or_nd, 1.5)

    # hysteresis thresholding
    edge_image = hys_thresh(I4, hi_thres, low_thres)

    h = hough_circles(edge_image, l_rad_sc, u_rad_sc)

    max_total = 0
    count = 0

    # find the maximum in the Hough space, and hence the parameters of the circle
    edge_height, edge_width = edge_image.shape
    for k in range(rd):
        for i in range(edge_width):
            for j in range(edge_height):
                cur = k * edge_width * edge_height + j * edge_width + i
                if h[cur] > max_total:
                    max_total = h[cur]
                    count = cur

    edge_n_data = edge_height * edge_width
    tmpi = count / edge_n_data

    r = int(((l_rad_sc + tmpi + 1) / scaling) + ADJ_PRECISION)
    tmpi = count - tmpi * edge_n_data

    row = tmpi // edge_width
    col = tmpi // edge_width
    row += 1
    col += 1

    row = int(row / scaling + ADJ_PRECISION)    # returns only first max value
    col = int(col / scaling + ADJ_PRECISION)

    # returning the result
    return row, col, r


#ToDo: Python code here. Optimize it with cython or anything like that.
#ToDo: The function can be optimized in less loops, but we might lose clarity
def canny(im, sigma, scaling, vert, horz):
    x_scaling = vert
    y_scaling = horz

    # obtaining image dimensions
    im_height, im_width = im.shape

    # applying gaussian blur
    size = int(6 * sigma + 1)
    new_im = cv2.GaussianBlur(im, (size, size), sigma)

    # resizing the image
    new_height = int(im_height * scaling)
    new_width = int(im_width * scaling)
    new_im = cv2.resize(new_im, (new_width, new_height))

    rows, cols = new_im.shape
    new_im = new_im.reshape(new_width * new_height)

    h = np.empty(rows * cols, np.float64)
    v = np.empty(rows * cols, np.float64)
    for i in range(rows):
        for j in range(cols):
            l_index = i * cols + j

            if j == 0:
                h[l_index] = new_im[i * cols + 1]
            elif j == cols - 1:
                h[l_index] = -new_im[i * cols + j - 1]
            else:
                h[l_index] = new_im[i * cols + j + 1] - new_im[i * cols + j - 1]

            # ---------------------------------

            if i == 0:
                v[l_index] = new_im[(i + 1) * cols + j]
            elif i == rows - 1:
                v[l_index] = -new_im[(i - 1) * cols + j]
            else:
                v[l_index] = new_im[(i + 1) * cols + j] - new_im[(i - 1) * cols + j]

    d1 = np.empty(rows * cols, np.float64)
    d2 = np.empty(rows * cols, np.float64)
    for i in range(rows):
        for j in range(cols):
            l_index = i * cols + j

            if i == rows - 1 or j == cols - 1:
                begin = 0
            else:
                begin = new_im[(i + 1) * cols + j + 1]

            if i == 0 or j == 0:
                end = 0
            else:
                end = new_im[(i - 1) * cols + j - 1]

            d1[l_index] = begin - end

            # ---------------------------------

            if i == 0 or j == cols - 1:
                begin = 0
            else:
                begin = new_im[(i - 1) * cols + j + 1]

            if i == rows - 1 or j == 0:
                end = 0
            else:
                end = new_im[(i + 1) * cols + j - 1]

            d2[l_index] = begin - end

    gradient = np.empty((rows, cols), np.float64)
    or_nd = np.empty((rows, cols), np.float64)

    for i in range(rows * cols):
        r = i // cols
        c = i % cols

        d1i = d1[i]
        d2i = d2[i]

        X = (h[i] + (d1i + d2i) / 2.0) * x_scaling
        Y = (v[i] + (d1i + d2i) / 2.0) * y_scaling

        gradient[r, c] = sqrt(X * X + Y * Y)

        or_nd_value = atan2(-Y, X)
        if or_nd_value < 0:
            or_nd_value += pi

        or_nd[r, c] = or_nd_value / pi * 180

    return gradient, or_nd


def adjust_gamma(im, g):
    # creting a copy of the image
    im_adjusted = im.copy()

    if g <= 0:
        return im_adjusted

    # rescale range 0-1
    min_val = im_adjusted.min()
    im_adjusted -= min_val

    max_val = im_adjusted.max()
    rows, cols = im_adjusted.shape
    for i in range(rows):
        for j in range(cols):
            im_adjusted[i, j] = pow(im_adjusted[i, j] / max_val, 1.0 / g)

    return im_adjusted


def non_max_suppression(in_image, orient, radius):
    # assuming all parameters are valid

    # getting image dimensions
    rows, cols = in_image.shape

    im = np.zeros((rows, cols), np.float64)
    i_radius = int(ceil(radius))

    hfrac = np.empty(FRAC_OFF_LENGTH, np.float64)
    vfrac = np.empty(FRAC_OFF_LENGTH, np.float64)
    xoff = np.empty(FRAC_OFF_LENGTH, np.float64)
    yoff = np.empty(FRAC_OFF_LENGTH, np.float64)

    # precalculate x and y offsets relative to centre pixel for each orientation angle
    for i in range(FRAC_OFF_LENGTH):   # 0 <= i <= 180
        angle = i * pi / 180                   # array of angles in 1 degree increments (but in radians).
        xoff[i] = radius * cos(angle)	       # x and y offset of points at specified radius and angle
        yoff[i] = radius * sin(angle)	       # from each reference position.
        hfrac[i] = xoff[i] - floor(xoff[i])    # fractional offset of xoff relative to integer location
        vfrac[i] = yoff[i] - floor(yoff[i])    # fractional offset of yoff relative to integer location

    yoff[180] = 0
    xoff[90] = 0

    # now run through the image interpolating grey values on each side of the centre pixel to be used for the
    # non-maximal suppression.
    for row in range(i_radius, rows - i_radius):
        for col in range(i_radius, rows - i_radius):
            ori = int(orient[row, col] + ADJ_PRECISION)     # index into precomputed arrays

            # x, y location on one side of the point in question
            x = col + xoff[ori]
            y = row - yoff[ori]

            # get integer pixel locations that surround location x,y
            fx = int(floor(x))
            cx = int(ceil(x))
            fy = int(floor(y))
            cy = int(ceil(y))

            tl = in_image[fy, fx]   # value at top left integer pixel location.
            tr = in_image[fy, cx]   # top right
            bl = in_image[cy, fx]   # bottom left
            br = in_image[cy, cx]   # bottom right

            upperavg = tl + hfrac[ori] * (tr - tl)  # now use bilinear interpolation to
            loweravg = bl + hfrac[ori] * (br - bl)  # estimate value at x,y
            v1 = upperavg + vfrac[ori] * (loweravg - upperavg)

            in_image_value = in_image[row, col]

            if in_image_value > v1:     # we need to check the value on the other side...

                x = col - xoff[ori]    # x, y location on the 'other side' of the point in question
                y = row + yoff[ori]

                fx = int(floor(x))
                cx = int(ceil(x))
                fy = int(floor(y))
                cy = int(ceil(y))

                tl = in_image[fy, fx]    # value at top left integer pixel location.
                tr = in_image[fy, cx]    # top right
                bl = in_image[cy, fx]    # bottom left
                br = in_image[cy, cx]    # bottom right

                upperavg = tl + hfrac[ori] * (tr - tl)
                loweravg = bl + hfrac[ori] * (br - bl)
                v2 = upperavg + vfrac[ori] * (loweravg - upperavg)

                if in_image_value > v2:          # this is a local maximum.
                    im[row, col] = in_image_value # record value in the output image.

    return im


def hys_thresh(im, t1, t2):
    # getting image dimensions
    rows, cols = im.shape
    n_data = rows * cols

    rc = n_data
    rcmr = rc - cols - 1
    rp1 = cols

    # creating the thresholded image
    im_thresholded = np.empty((rows, cols), np.uint8)

    # find indices of all pixels with value > T1 and the amount
    bw = im.reshape(n_data)
    pix = np.empty(n_data, np.float64)
    npix = 0
    for i in range(n_data):
        if bw[i] > t1:
            pix[npix] = i
            npix += 1

    # create a stack array (that should never overflow!
    stack = np.zeros(n_data, np.int32)

    # setting stack pointer (stp)
    stp = npix - 1
    for i in range(npix):
        stack[i] = pix[i]   # put all the edge points on the stack
        bw[pix[i]] = -1     # mark points as edges

    # precompute an array, O, of index offset values that correspond to the eight
    # surrounding pixels of any point. Note that the image was transformed into
    # a column vector, so if we reshape the image back to a square the indices
    # surrounding a pixel with index, n, will be:
    #       n-cols-1   n-1   n+cols-1
    #
    #       n-cols     n     n+cols
    #
    #       n-cols+1   n+1   n+cols+1

    tmp = np.empty(8, np.int32)
    index = np.empty(8, np.int32)

    tmp[0] = -1
    tmp[1] = 1
    tmp[2] = -cols - 1
    tmp[3] = -cols
    tmp[4] = -cols + 1
    tmp[5] = cols - 1
    tmp[6] = cols
    tmp[7] = cols + 1

    # while the stack is not empty
    while stp >= 0:
        v = stack[stp]      # pop next index off the stack
        stp -= 1

        # prevent us from generating illegal indices
        if rp1 < v < rcmr:
            # now look at surrounding pixels to see if they should be pushed onto the
            # stack to be processed as well.

            # calculate indices of points around this pixel.
            for i in range(8):
                index[i] = tmp[i] + v

            for l in range(8):
                ind = index[l]

                if bw[ind] > t2:
                    stp += 1            # push index onto the stack.
                    stack[stp] = ind
                    bw[ind] = -1        # mark this as an edge point

    # finally zero out anything that was not an edge
    for i in range(n_data):
        if bw[i] == -1:
            bw[i] = 1
        else:
            bw[i] = 0

    # reshaping bw
    bw = bw.reshape(rows, cols)

    # returning bw
    return bw


def hough_circles(m_edge_im, r_min, r_max):
    # getting image dimenssions
    edge_im = m_edge_im.copy()
    rows, cols = m_edge_im.shape
    n_radii = r_max - r_min + 1

    # creating the accumulator
    h = np.zeros(rows * cols * n_radii, np.float64)

    x = np.empty(rows * cols, np.float64)
    y = np.empty(rows * cols, np.float64)

    y_size = 0
    for i in range(rows):
        for j in range(cols):
            if edge_im[i, j] != 0:
                x[y_size] = i
                y[y_size] = j
                y_size += 1

    for index in range(y_size):
        cx = x[index]
        cy = y[index]

        for n in range(n_radii):
            add_circle(h, n * rows * cols, rows, cols, cx + 1, cy + 1, n + 1 + r_min)

    return h


# NOTE: This function modifies h
def add_circle(h, h_bptr, hr, hc, cx, cy, radius):
    tmp = radius / 1.4142
    fix_radius = int(tmp)

    px = np.empty((fix_radius + 1) * 8, np.int32)
    py = np.empty((fix_radius + 1) * 8, np.int32)

    valid_px = np.empty((fix_radius + 1) * 8, np.int32)
    valid_py = np.empty((fix_radius + 1) * 8, np.int32)
    ind = np.empty((fix_radius + 1) * 8, np.int32)
    mark = np.zeros(hr * hc, np.int32)

    weight = 1
    valid = 0

    for i in range(fix_radius + 1):
        x = i
        tmp = x * x
        tmp = tmp / (radius * radius)
        cos_theta = sqrt(1 - tmp)
        y = int(radius * cos_theta + 0.5)

        px[i] = cy + x
        px[i + fix_radius + 1] = cy + y
        px[i + 2 * (fix_radius + 1)] = cy + y
        px[i + 3 * (fix_radius + 1)] = cy + x
        px[i + 4 * (fix_radius + 1)] = cy - x
        px[i + 5 * (fix_radius + 1)] = cy - y
        px[i + 6 * (fix_radius + 1)] = cy - y
        px[i + 7 * (fix_radius + 1)] = cy - x

        py[i] = cx + y
        py[i + fix_radius + 1] = cx + x
        py[i + 2 * (fix_radius + 1)] = cx - x
        py[i + 3 * (fix_radius + 1)] = cx - y
        py[i + 4 * (fix_radius + 1)] = cx - y
        py[i + 5 * (fix_radius + 1)] = cx - x
        py[i + 6 * (fix_radius + 1)] = cx + x
        py[i + 7 * (fix_radius + 1)] = cx + y

    for i in range(8 * fix_radius + 1):
        px_i = px[i]
        py_i = py[i]
        if 1 <= px_i <= hr and 1 <= py_i <= hc:
            valid_px[valid] = px_i
            valid_py[valid] = py_i
            ind[valid] = (valid_px[valid] - 1) * hc + valid_py[valid]
            l_index = ind[valid] - 1

            if not mark[l_index]:
                h[h_bptr + l_index] += weight
                mark[l_index] = 1
                valid += 1


def round_nd(x):
    return int(round(x))
