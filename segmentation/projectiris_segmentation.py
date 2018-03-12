import cv2
import numpy as np

from utils.math_utils import compute_circle_center_coords, euclidean_distance_coords
from utils.error_utils import SUCCESS, WRONG_IMAGE_FORMAT, PUPIL_DETECTION_FAILED, IRIS_DETECTION_FAILED, EYELIDS_DETECTION_FAILED

#--------------------------------------------------------------------------------

#this means that there's a scale difference between the filter size of the median filter
#implemented in the project iris and the one in OpenCV
#ex: project_iris_median(3) = opencv_median(9)      9/3 = 3
#    project_iris_median(9) = opencv_median(27)     27/9 = 3
KERNEL_SIZE_SCALE = 3

#--------------------------------------------------------------------------------

FIRST_MAX_UPPER_BOUND = 170
FIRST_MAX_LOWER_BOUND = 131
FIRST_MAX_DEFAULT_VALUE = 160
FIRST_MAX_OFFSET = 13

#--------------------------------------------------------------------------------

BLACK = 0
WHITE = 255

#--------------------------------------------------------------------------------

PUPIL_THRESHOLD = 0
PUPIL_CIRCLE_THICKNESS = 2
PUPIL_MIN_RADIUS = 10
PUPIL_MAX_RADIUS = 60
PUPIL_RADIUS_INC = 1
PUPIL_CENTER_OFFSET = 5

#--------------------------------------------------------------------------------

IRIS_RADIUS_INC = -5

#--------------------------------------------------------------------------------

BLACK_THRESHOLD = 80

#--------------------------------------------------------------------------------

EYELIDS_POINTS_DISTANCE = 50     # this is the distance between the eyelids points
EYELIDS_PADDING = 21             # i have no idea what it means
EYELID_POINTS_COUNT = 3          # amount of points of an eyelid
#--------------------------------------------------------------------------------


def segment_iris(eye_img):
    #image must be in grayscale
    if eye_img.dtype != np.uint8:
        return WRONG_IMAGE_FORMAT, None

    # ----------------------------------------------------------------------------

    #trying to detect pupil (from original image)
    pupil_data = find_pupil(eye_img)
    if pupil_data is None:
        return PUPIL_DETECTION_FAILED, None

    # ----------------------------------------------------------------------------

    #detecting iris
    iris_data = find_iris(eye_img, pupil_data[0])  # pupil_data[0] = pupil center
    if iris_data is None:
        return IRIS_DETECTION_FAILED, (pupil_data, None, None)

    # ----------------------------------------------------------------------------

    #detecting eyelids
    eyelids_data = find_eyelids(eye_img, pupil_data[0])
    if eyelids_data is None:
        return EYELIDS_DETECTION_FAILED, (pupil_data, iris_data, None)

    # ----------------------------------------------------------------------------

    #all went well
    return SUCCESS, (pupil_data, iris_data, eyelids_data)


#return the center and the radius
def find_pupil(img):
    #creating a copy of the original image
    copy = img.copy()

    #applying median filter
    cv2.medianBlur(copy, 9 * KERNEL_SIZE_SCALE, copy)

    #computing pupil threshold
    PUPIL_THRESHOLD = get_pupil_threshold(img)

    #applying binary threshold
    cv2.threshold(copy, PUPIL_THRESHOLD, WHITE, cv2.THRESH_BINARY, copy)

    #ToDo: Python code here. Optimize it with cython or anything like that.
    sumx = 0
    sumy = 0
    amount = 0

    height, width = copy.shape
    for x in range(width):
        for y in range(height):
            #black pixel
            if not copy.item(y, x):
                sumx += x
                sumy += y
                amount += 1

    # If sumx and sumy are 0, that means that the filter destroyed the pupil, so
    # autodetection failed
    if sumx == 0 or sumy == 0:
        return None

    sumx //= amount
    sumy //= amount

    radius = 0
    i = sumy
    j = sumx
    # starting from the center and going right
    while not copy.item(i, j):
        radius += 1
        j += 1

    # 2 of padding
    radius -= 2
    radius = max(0, radius)

    #ToDo: Read below.
    # Here i'm going to draw a circle instead of applying sobel again. Instead of
    # detecting the circle in the image i think i should resize the image to a
    # smaller size so the process of finding a circle is less computationally
    # expensive

    white_img = np.ones(copy.shape, np.uint8) * 255
    cv2.circle(white_img, (sumx, sumy), radius, BLACK, PUPIL_CIRCLE_THICKNESS)

    #ToDo: Optimize here using Circle Hough Transform from OpenCV and not this function
    center_rect = (sumx - 1, sumx + 1, sumy - 1, sumy + 1)
    pupil_data = find_circle(white_img, center_rect, radius, radius + 4)

    #if find_circle had some troubles
    if pupil_data is None:
        return None

    p1, p2, p3 = pupil_data
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    xc, yc = compute_circle_center_coords(x1, y1, x2, y2, x3, y3)
    pupil_radius = euclidean_distance_coords(xc, yc, x1, y1)  # could have been p2 or p3

    return (xc, yc), int(pupil_radius + PUPIL_RADIUS_INC)


#the img is the binary thing, not the original
def find_iris(img, pupil_center):
    xc, yc = pupil_center
    center_rect = (xc - PUPIL_CENTER_OFFSET, xc + PUPIL_CENTER_OFFSET,
                   yc - PUPIL_CENTER_OFFSET, yc + PUPIL_CENTER_OFFSET)
    height, width = img.shape

    iris_data = find_circle(img, center_rect, width // 4, height // 2)

    #if find_circle had some troubles
    if iris_data is None:
        return None

    p1, p2, p3 = iris_data
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    xc, yc = compute_circle_center_coords(x1, y1, x2, y2, x3, y3)
    iris_radius = euclidean_distance_coords(xc, yc, x1, y1)  # could have been p2 or p3

    return (xc, yc), int(iris_radius + IRIS_RADIUS_INC)


#This is practically the same implementation of project iris
def find_eyelids(img, pupil_center):
    #performing the "otsu" thresholding
    eyelids = img.copy()

    #performing median blur
    cv2.medianBlur(src=eyelids, ksize=9 * KERNEL_SIZE_SCALE, dst=eyelids)

    #performing threshold with the "no black" index
    no_black_threshold = get_threshold_without_black(eyelids)
    cv2.threshold(src=eyelids, thresh=no_black_threshold, maxval=WHITE, type=cv2.THRESH_BINARY, dst=eyelids)

    #performing median blur again
    eyelids = cv2.medianBlur(src=eyelids, ksize=9 * KERNEL_SIZE_SCALE, dst=eyelids)

    #finding eyelids
    xc, _ = pupil_center               # pupil_center is a tuple with 2 elements
    width = xc
    height = eyelids.shape[0] // 2 - 1   # shape[0] is the amount of rows of the array
    point_dist = EYELIDS_POINTS_DISTANCE
    padding = EYELIDS_PADDING

    upper_eyelid = [None, None, None]
    lower_eyelid = [None, None, None]

    #ToDo: Python code here. Optimize it with cython or anything like that.
    for i in range(height):

        # First eyelid (top-down)
        for j in range(EYELID_POINTS_COUNT):
            dist_along = width - point_dist + (j * point_dist)
            if not upper_eyelid[j] and eyelids.item(i, dist_along + 50) != WHITE:
                upper_eyelid[j] = (dist_along, i + padding)

        # second eyelid (bottom-up)
        for j in range(EYELID_POINTS_COUNT):
            dist_along = width - point_dist + (j * point_dist)
            if not lower_eyelid[j] and eyelids.item(2 * height - i, dist_along) != WHITE:
                lower_eyelid[j] = (dist_along, 2 * height - i)

    #cheking if eyelids were found
    for i in range(EYELID_POINTS_COUNT):
        if not upper_eyelid[i] or not lower_eyelid[i]:
            return None

    #assuming it always went well
    return tuple(upper_eyelid), tuple(lower_eyelid)


#ToDo: This is a Circle Hough Transform implementation. Use the one in OpenCV
#ToDo: Python code here. Optimize it with cython or anything like that.
def find_circle(img, centerRegion, minRadius, maxRadius):
    #centerRegion is the rectangle for iris center

    if centerRegion is None:
        height, width = img.shape
        # (x_left, x_right, y_top, y_bottom) ->the whole image
        centerRegion = (0, width - 1, 0, height - 1)

    x_left, x_right, y_top, y_bottom = centerRegion
    a_min = x_left
    a_max = x_right
    b_min = y_top
    b_max = y_bottom

    r_min = minRadius
    r_max = maxRadius

    a = a_max - a_min
    b = b_max - b_min
    r = r_max - r_min

    # The sum of the values which are >= maxVotes - 1
    total_r = 0
    total_a = 0
    total_b = 0

    # Amount of values that are >= maxVotes - 1
    amount = 0

    # The max amount of votes that any has
    maxVotes = 0

    # Create and initialise accumulator to 0
    acc = np.zeros((a, b, r), np.int32)

    # For each black point, find the circles which satisfy the equation where the
    # parameters are limited by a,b and r.
    height, width = img.shape
    for x in range(width):
        for y in range(height):
            #if pixel is white continue
            if img.item(y, x):
                continue

            for _a in range(a):
                for _b in range(b):
                    for _r in range(r):
                        sq_a = x - (_a + a_min)
                        sq_b = y - (_b + b_min)
                        sq_r = r - (_r + r_min)

                        if sq_a * sq_a + sq_b * sq_b == sq_r * sq_r:
                            new_value = acc.item((_a, _b, _r)) + 1
                            acc.itemset((_a, _b, _r), new_value)

                            if new_value >= maxVotes:
                                maxVotes = new_value

    for _a in range(a):
        for _b in range(b):
            for _r in range(r):
                if acc.item((_a, _b, _r)) >= maxVotes - 1:
                    total_a += _a + a_min
                    total_b += _b + b_min
                    total_r += _r + r_min
                    amount += 1

    # Get the initial average values
    top_a = total_a / amount
    top_b = total_b / amount
    top_r = total_r / amount

    # Returning the three points
    p1 = (top_a + top_r, top_b)
    p2 = (top_a - top_r, top_b)
    p3 = (top_a, top_b + top_r)

    return p1, p2, p3


#ToDo: Python code here. Optimize it with cython or anything like that.
def get_pupil_threshold(img):
    hist = build_histogram(img)
    pupil_max = -1
    pupil_max_index = -1

    for i in range(90):
        current_value = hist.item(i)

        if current_value > pupil_max:
            pupil_max = current_value
            pupil_max_index = i

    return pupil_max_index + 8


#ToDo: Python code here. Optimize it with cython or anything like that.
def get_iris_threshold(img):
    first_max_index = -1
    second_max_index = -1
    min_index = -1

    #The histogram generally shows 4 peaks as a property of the image. The darkest peak
    #represents the mass of dark pixels in the pupil, the next lightest peak is usually
    #the overall mass of pixels in the iris. It is usually a good idea to apply the median
    #filter first to the image to extract the best peaks out of the image to avoid
    #variation from noise.
    cv2.medianBlur(img, 9 * KERNEL_SIZE_SCALE, img)

    #building a smoooth histogram
    smooth_hist = build_smooth_histogram(img)

    #Plotted on a graph, we should have some distinct peaks, two for the greyish centres. The
    #first, and highest peak is what we want to threshold around so we drop the 8 outermost
    #groups and find the maximum value.
    first_max = -1
    for i in range(90, 240):
        current_value = smooth_hist.item(i)

        if current_value > first_max:
            first_max = current_value
            first_max_index = i

    #Now find the second maximum so that we can threshold around this central point.
    #Possibly more accurate.
    second_max = -1

    #first search to the right
    for i in range(first_max_index + 20, 240):
        current_value = smooth_hist.item(i)

        if current_value > second_max:
            second_max = current_value
            second_max_index = i

    #now search to the left (change)
    for i in range(91, (first_max_index - 20) + 1):
        current_value = smooth_hist.item(i)

        if current_value > second_max:
            second_max = current_value
            second_max_index = i

    #Now find the minimum between these two.
    minimum = first_max

    #swap them if necessary
    #ToDo: Fix. Non-pythonic code.
    if first_max_index > second_max_index:
        tmp = first_max_index
        first_max_index = second_max_index
        second_max_index = tmp

    for i in range(first_max_index, second_max_index):
        current_value = smooth_hist.item(i)

        if current_value < minimum:
            minimum = current_value
            min_index = i

    return  first_max_index, second_max_index, min_index


def build_histogram(img):
    return cv2.calcHist([img], [0], None, [256], [0, 256])


#ToDo: Python code here. Optimize it by using cython or anything like that.
def build_smooth_histogram(img):
    hist = build_histogram(img)
    smooth_hist = np.array([0] * 256, np.int32)

    smooth_hist.itemset(0, (hist.item(0) + hist.item(1)) / 2)

    for i in range(1, 255):
        value = (hist.item(i - 1) + hist.item(i) + hist.item(i + 1)) / 3
        smooth_hist.itemset(i, value)

    smooth_hist.itemset(255, (hist.item(254) + hist.item(255)) / 2)

    return smooth_hist


#we assume that src has the same size of other, and both are of dtype = uint8
def max_blend(src, other):
    result = np.empty(src.shape, np.uint8)
    cv2.max(src, other, result)
    return result


#we assume that src has the same size of other, and both are of dtype = uint8
def min_blend(src, other):
    result = np.empty(src.shape, np.uint8)
    cv2.min(src, other, result)
    return result


def add_blend(src, other):
    #ToDo: I'm going to do it by hand for now. Find how to do it with OpenCV
    height, width = src.shape
    result = np.empty(src.shape, np.uint8)

    for i in range(height):
        for j in range(width):
            pix_value = src.item(i, j) + other.item(i, j)

            if pix_value > 255:
                pix_value = 255

            #ToDo: I think it could be removed, because the sum will always be > 0
            elif pix_value < 0:
                pix_value = 0

            result.itemset(i, j, 255 - pix_value)

    return result


#ToDo: Python code here. Optimize it by using cython or anything like that.
def get_threshold_without_black(img):
    #computing histogram
    hist = build_histogram(img)

    #computing number of pixels with value >= 80
    pixel_count = 0
    for i in range(BLACK_THRESHOLD, 256):
        pixel_count += hist.item(i)

    #computing the probability distribution from histogram
    p = compute_probability_distribution(hist, pixel_count)

    #returning the max goodness index
    return get_max_goodness_index(p)


#ToDo: Python code here. Optimize it by using cython or anything like that.
#ToDo: Do all in a single pass-omega, mew, etc- (i want this code to resemble the project iris source code)
def get_max_goodness_index(prob_dist):
    #compute goodness array
    goodness = np.empty(256, np.float32)
    mew_256 = compute_mew(256, prob_dist)
    for k in range(256):
        wk = compute_omega(k, prob_dist)
        mk = compute_mew(k, prob_dist)

        if wk == 0 or wk == 1:
            goodness_value = 0
        else:
            factor = mew_256 * wk - mk
            num = factor * factor
            den = wk * (1.0 - wk)
            goodness_value = num / den

        goodness.itemset(k, goodness_value)

    #get max index (this can be done in the previous step)
    max_index = 0
    for i in range(256):
        if goodness.item(i) > goodness.item(max_index):
            max_index = i

    return max_index


#ToDo: Python code here. Optimize it by using cython or anything like that.
def compute_probability_distribution(hist, total_pixels):
    prob_dist = np.empty(256, dtype=np.float32)
    total = float(total_pixels)

    for i in range(256):
        if i < BLACK_THRESHOLD:
            prob_dist.itemset(i, 0.0)
        else:
            prob_dist.itemset(i, hist.item(i) / total)

    return prob_dist


#ToDo: Python code here. Optimize it by using cython or anything like that.
def compute_omega(k, prob_dist):
    result = 0.0

    for i in range(k):
        result += prob_dist.item(i)

    return result


#ToDo: Python code here. Optimize it by using cython or anything like that.
def compute_mew(k, prob_dist):
    result = 0.0

    for i in range(k):
        result += (i + 1) * prob_dist.item(i)

    return result
